/* The GS renderer on the PS4 (port/ps4/PLAN.md, phase 4): the exact front end gpu_exact.inc over a
   CPU device. The PS4 has neither Vulkan nor Metal nor Direct3D, so the Rust/wgpu renderer
   (port/gs/gpu_rust.c) does not build there; this file is its device, done by the console's cores.

   - The device's buffers are memory (B_VRAM, the texture snapshot, the readback, staging and upload
     pages); the front end's arenas (states, primitives, tile lists, batch records) are read where
     the front end keeps them, so their queue writes copy nothing.
   - Commands (copies, batches, the local copy, the scanout) wait in a list and run at op_submit, in
     order, as a GPU's command buffer would: a submit's queue writes land before its commands.
   - raster.wgsl's three entry points are translated to C below: main (draw_tile, a batch's tiles
     shared out over worker threads; each 8x8 tile is independent, as one workgroup is), local_copy
     and scanout. The translation keeps its arithmetic: float math in the same order (no FMA:
     FP_CONTRACT off), D3D's float-to-integer rules, and the depth plane in double precision, which
     the shader emulates (sd_*) and the CPU has.
   - The picture: the scanout (or a movie's pixels) is scaled to the 1920x1080 output (4:3 with
     side bars, or the widescreen setting's shape) into write-combined sceVideoOut buffers, as the
     Cave Story port does, and flipped at the next vertical blank. */
#pragma STDC FP_CONTRACT OFF
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <orbis/SystemService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include "../gs/gs.h"
#include "../gs/gpu.h"
#include "../gs/perf_log.h"
#include "gnm.h"

double host_widescreen_aspect(void);  /* port/hle/widescreen.c: the picture's shape, 0 for 4:3 */

/* Batches of up to 2048 primitives (the shared front end's default is 256): a batch is one GPU
   dispatch, followed by a barrier, so larger ones keep more of the GPU busy and wait less. */
#define EXACT_MAX_DRAWS 2048
#include "../gs/gpu_exact.inc"  /* the front end: Exact, the CPU half, and the op_* prototypes it calls */

/* ---- the device ---- */

enum { CMD_COPY, CMD_SNAPSHOT, CMD_BATCH, CMD_LOCAL_COPY, CMD_SCANOUT };
typedef struct {
    int kind;
    int from, to;
    uint32_t from_offset, to_offset, bytes;  /* CMD_COPY */
    uint32_t pages[16];                      /* CMD_SNAPSHOT: the VRAM pages that changed since the last */
    uint32_t offset;                         /* CMD_BATCH: the batch record */
    uint32_t c[12];                          /* CMD_LOCAL_COPY, CMD_SCANOUT: their constants */
} Command;

typedef struct Cpu {
    Exact e;                     /* first: g->backend points at it */
    uint32_t *vram, *snapshot, *readback, *staged, *upload;
    uint32_t display_constants[12], copy_constants[12];
    Command *commands;
    unsigned command_count, command_capacity;
    uint32_t *screen;            /* the scanout, RGBA (red in the low byte), screen_width x screen_height */
    uint32_t *field;             /* field mode: one field's lines before they are doubled (run_scanout) */
    unsigned screen_width, screen_height;
    uint64_t tiles, batches, raster_ticks, snapshot_pages, snapshots;
} Cpu;

static uint8_t *buffer_base(Cpu *cpu, int buffer)
{
    switch (buffer) {
    case B_VRAM: return (uint8_t *)cpu->vram;
    case B_SNAPSHOT: return (uint8_t *)cpu->snapshot;
    case B_STATES: return (uint8_t *)cpu->e.state_data;
    case B_PRIMITIVES: return (uint8_t *)cpu->e.primitive_data;
    case B_BINS: return (uint8_t *)cpu->e.bin_data;
    case B_DISPLAY: return (uint8_t *)cpu->display_constants;
    case B_COPY: return (uint8_t *)cpu->copy_constants;
    case B_READBACK: return (uint8_t *)cpu->readback;
    case B_STAGED: return (uint8_t *)cpu->staged;
    case B_BATCHES: return (uint8_t *)cpu->e.batch_data;
    case B_UPLOAD: return (uint8_t *)cpu->upload;
    default: return NULL;
    }
}

static Command *command_add(Cpu *cpu, int kind)
{
    if (cpu->command_count == cpu->command_capacity) {
        unsigned capacity = cpu->command_capacity ? cpu->command_capacity * 2 : 1024;
        Command *grown = realloc(cpu->commands, capacity * sizeof *grown);
        if (!grown) { fprintf(stderr, "gs: out of memory for the CPU device's commands\n"); abort(); }
        cpu->commands = grown;
        cpu->command_capacity = capacity;
    }
    Command *c = &cpu->commands[cpu->command_count++];
    memset(c, 0, sizeof *c);
    c->kind = kind;
    return c;
}

/* ---- raster.wgsl in C (raster_core.h) ---- */

#define RC_MEM
#define RC_LOAD(p) __atomic_load_n((p), __ATOMIC_RELAXED)
#define RC_STORE(p, v) __atomic_store_n((p), (v), __ATOMIC_RELAXED)
static inline uint32_t rc_cas(uint32_t *word, uint32_t expected, uint32_t desired)
{
    __atomic_compare_exchange_n(word, &expected, desired, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}
#define RC_CAS(p, expected, desired) rc_cas((p), (expected), (desired))
static inline int32_t rc_abs(int32_t v) { return v < 0 ? -v : v; }
#include "raster_core.h"

/* raster.wgsl's main for one touched tile: its primitives in submission order, each over the
   tile's pixels inside its box (pixels of one batch are independent, so the order between pixels
   is free, as between a workgroup's invocations). */
static void draw_tile(Cpu *cpu, uint32_t entry)
{
    const uint32_t *bins = cpu->e.bin_data;
    uint32_t tile = bins[entry], first = bins[entry + 1], count = bins[entry + 2];
    uint32_t x0 = (tile & 0xFFFFu) * 8u, y0 = (tile >> 16) * 8u;
    for (uint32_t i = 0; i < count; i++) {
        const Primitive *d = &cpu->e.primitive_data[bins[first + i]];
        const Constants *st = &cpu->e.state_data[d->info[3]];
        Setup setup;
        int32_t box[4];
        if (!prim_box(d, x0, y0, &setup, box)) continue;
        for (int32_t y = box[1]; y < box[3]; y++)
            for (int32_t x = box[0]; x < box[2]; x++)
            {
                PixelCache cache = {{0, 0}, {0, 0}};
                draw_pixel(cpu->vram, cpu->snapshot, d, st, &setup, &cache, (uint32_t)x, (uint32_t)y);
            }
    }
}

/* ---- the worker threads: a batch's tiles shared out ---- */

#define MAX_WORKERS 8
static struct {
    pthread_mutex_t lock;
    pthread_cond_t start, finished;
    unsigned generation, running, workers;
    Cpu *cpu;
    uint32_t base, tiles;
    unsigned next;               /* the next tile to take (atomic) */
} pool = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0, 0, NULL, 0, 0, 0};

enum { TILES_PER_TAKE = 4 };

/* A run of batches (run_batches): each thread draws, batch after batch, the tiles it owns. */
static struct { const Command *first; unsigned count, threads; } run;
static unsigned tile_owner(uint32_t tile, unsigned threads)
{
    uint32_t tx = tile & 0xFFFFu, ty = tile >> 16;
    return ((tx >> 1) + (ty >> 1) * 3u) % threads;
}
static void draw_owned(unsigned me)
{
    Cpu *cpu = pool.cpu;
    const uint32_t *bins = cpu->e.bin_data;
    for (unsigned b = 0; b < run.count; b++) {
        const uint32_t *header = cpu->e.batch_data + run.first[b].offset / 4;
        uint32_t tiles = header[0], base = header[1];
        for (uint32_t n = 0; n < tiles; n++) {
            uint32_t entry = base + n * 3u;
            if (tile_owner(bins[entry], run.threads) == me) draw_tile(cpu, entry);
        }
    }
}

static void take_tiles(void)
{
    for (;;) {
        unsigned n = __atomic_fetch_add(&pool.next, TILES_PER_TAKE, __ATOMIC_RELAXED);
        if (n >= pool.tiles) return;
        unsigned end = n + TILES_PER_TAKE < pool.tiles ? n + TILES_PER_TAKE : pool.tiles;
        for (; n < end; n++) draw_tile(pool.cpu, pool.base + n * 3u);
    }
}

static void *worker(void *arg)
{
    unsigned me = (unsigned)(uintptr_t)arg;  /* 1..workers; the GS thread is 0 */
    unsigned seen = 0;
    pthread_mutex_lock(&pool.lock);
    for (;;) {
        while (pool.generation == seen) pthread_cond_wait(&pool.start, &pool.lock);
        seen = pool.generation;
        int owned = run.count != 0;
        pthread_mutex_unlock(&pool.lock);
        if (owned) draw_owned(me);
        else take_tiles();
        pthread_mutex_lock(&pool.lock);
        if (--pool.running == 0) pthread_cond_signal(&pool.finished);
    }
    return NULL;
}

static void pool_start(void)
{
    const char *setting = getenv("DHD_CPU_THREADS");
    unsigned workers = setting ? (unsigned)atoi(setting) : 4u;
    if (workers > MAX_WORKERS) workers = MAX_WORKERS;
    for (unsigned i = 0; i < workers; i++) {
        pthread_t thread;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 256 * 1024);
        if (pthread_create(&thread, &attr, worker, (void *)(uintptr_t)(pool.workers + 1)) == 0) {
            pthread_detach(thread);
            pool.workers++;
        }
        pthread_attr_destroy(&attr);
    }
    fprintf(stderr, "gs: CPU renderer, %u worker threads beside the GS thread\n", pool.workers);
}

/* One batch: few tiles on this thread, more shared with the workers (this thread takes some too). */
static void run_batch(Cpu *cpu, uint32_t offset)
{
    const uint32_t *header = cpu->e.batch_data + offset / 4;
    uint32_t tiles = header[0], base = header[1];
    cpu->tiles += tiles;
    cpu->batches++;
    if (tiles < 8 || !pool.workers) {
        for (uint32_t n = 0; n < tiles; n++) draw_tile(cpu, base + n * 3u);
        return;
    }
    pthread_mutex_lock(&pool.lock);
    run.count = 0;
    pool.cpu = cpu; pool.base = base; pool.tiles = tiles;
    __atomic_store_n(&pool.next, 0, __ATOMIC_RELAXED);
    pool.running = pool.workers;
    pool.generation++;
    pthread_cond_broadcast(&pool.start);
    pthread_mutex_unlock(&pool.lock);
    take_tiles();
    pthread_mutex_lock(&pool.lock);
    while (pool.running) pthread_cond_wait(&pool.finished, &pool.lock);
    pthread_mutex_unlock(&pool.lock);
}

/* A batch whose primitives all have independent pixels and draw on one frame and Z buffer
   (key): batches in a row with the same key touch the same words from the same tiles only, so
   each thread can draw its own tiles through all of them in order: one hand-out for the run, not
   one per batch (a menu field has some 250 batches). 0 when the batch must run alone. */
static uint64_t batch_key(Cpu *cpu, uint32_t offset)
{
    const uint32_t *header = cpu->e.batch_data + offset / 4;
    const uint32_t *bins = cpu->e.bin_data;
    uint32_t tiles = header[0], base = header[1];
    uint64_t key = 0;
    for (uint32_t n = 0; n < tiles; n++) {
        uint32_t first = bins[base + n * 3u + 1], count = bins[base + n * 3u + 2];
        for (uint32_t i = 0; i < count; i++) {
            const Primitive *d = &cpu->e.primitive_data[bins[first + i]];
            const Constants *st = &cpu->e.state_data[d->info[3]];
            uint32_t ctxt = (d->info[1] >> 9) & 1u;
            if (!(st->info[1] & 1u)) return 0;
            uint64_t k = ((uint64_t)(REG_X(st, 76 + ctxt) & 0x3F3F01FFu) << 32) |
                         (REG_X(st, 78 + ctxt) & 0x0F0001FFu) | (1u << 31);
            if (key && k != key) return 0;
            key = k;
        }
    }
    return key;
}

/* count batches in a row with one key, shared out by tile owner. */
static void run_batches(Cpu *cpu, const Command *first, unsigned count)
{
    for (unsigned b = 0; b < count; b++) {
        cpu->tiles += cpu->e.batch_data[first[b].offset / 4];
        cpu->batches++;
    }
    if (!pool.workers) {
        for (unsigned b = 0; b < count; b++) {
            const uint32_t *header = cpu->e.batch_data + first[b].offset / 4;
            for (uint32_t n = 0; n < header[0]; n++) draw_tile(cpu, header[1] + n * 3u);
        }
        return;
    }
    pthread_mutex_lock(&pool.lock);
    run.first = first; run.count = count; run.threads = pool.workers + 1;
    pool.cpu = cpu;
    pool.running = pool.workers;
    pool.generation++;
    pthread_cond_broadcast(&pool.start);
    pthread_mutex_unlock(&pool.lock);
    draw_owned(0);
    pthread_mutex_lock(&pool.lock);
    while (pool.running) pthread_cond_wait(&pool.finished, &pool.lock);
    run.count = 0;
    pthread_mutex_unlock(&pool.lock);
}

/* raster.wgsl's local_copy: one rectangle to another (on different pages). */
static void run_local_copy(Cpu *cpu, const uint32_t c[12])
{
    for (uint32_t y = 0; y < c[9]; y++)
        for (uint32_t x = 0; x < c[8]; x++) {
            uint32_t v = read_pixel(cpu->vram, c[0], c[1], c[2], (c[3] + x) & 2047u, (c[4] + y) & 2047u);
            write_pixel(cpu->vram, c[5], c[6], c[7], (c[10] + x) & 2047u, (c[11] + y) & 2047u, v, 0u);
        }
}

/* raster.wgsl's scanout: the display circuit's picture (VRAM itself) into the screen image. */
/* raster.wgsl's scanout: the display circuit's picture (VRAM itself) into the screen image.

   Field mode (c[7]: interlaced, one field's lines in the frame buffer, each shown twice): the game
   draws each field half a line apart, for a TV's interlace, which shows the fields' lines between
   each other. A progressive screen shows that as the picture jumping a line up and down every
   field. So a field is shown one line lower when that matches the picture before it better
   (sampled differences of the green channel; a tie keeps the last choice), which keeps it still. */
/* The scanout's source lines: a field's (c[7]) or the whole picture's. */
static uint32_t scan_lines(const uint32_t c[12]) { return c[7] ? (c[6] + 1) / 2 : c[6]; }

/* The screen image from the scanned lines in cpu->field (see run_scanout). */
static void compose_scanout(Cpu *cpu, const uint32_t c[12], uint32_t width, uint32_t height)
{
    static int shift;  /* the last field's: 0, or 1 = shown one line lower */
    uint32_t lines = scan_lines(c);
    if (!c[7]) {
        memcpy(cpu->screen, cpu->field, (size_t)width * height * 4);
        cpu->screen_width = width;
        cpu->screen_height = height;
        return;
    }
    if (cpu->screen_width == width && cpu->screen_height == height && height >= 32) {
        uint64_t cost[2] = {0, 0};
        for (int s = 0; s < 2; s++)
            for (uint32_t y = 8; y + 8 < height; y += 4) {
                /* shown one line lower: screen row y shows field line (y - 1) / 2 */
                uint32_t line = s ? (y - 1) / 2 : y / 2;
                const uint32_t *f = cpu->field + (size_t)line * width, *old = cpu->screen + (size_t)y * width;
                for (uint32_t x = 4; x < width; x += 16) {
                    int g = (int)((f[x] >> 8) & 255u) - (int)((old[x] >> 8) & 255u);
                    cost[s] += (uint64_t)(g < 0 ? -g : g);
                }
            }
        if (cost[0] != cost[1]) shift = cost[1] < cost[0];
    }
    for (uint32_t y = 0; y < height; y++) {
        uint32_t line = shift ? (y ? (y - 1) / 2 : 0) : y / 2;
        if (line >= lines) line = lines - 1;
        memcpy(cpu->screen + (size_t)y * width, cpu->field + (size_t)line * width, width * 4);
    }
    cpu->screen_width = width;
    cpu->screen_height = height;
}

static void run_scanout(Cpu *cpu, const uint32_t c[12], uint32_t width, uint32_t height)
{
    uint32_t lines = scan_lines(c);
    for (uint32_t y = 0; y < lines; y++) {
        uint32_t *row = cpu->field + (size_t)y * width;
        for (uint32_t x = 0; x < width; x++) row[x] = scan_pixel(cpu->vram, c, x, y);
    }
    compose_scanout(cpu, c, width, height);
}

/* ---- the device ops ---- */

static void op_write(Exact *gpu, int buffer, uint32_t offset, const void *data, uint32_t bytes)
{
    uint8_t *to = buffer_base((Cpu *)gpu, buffer);
    if (to && to + offset != data) memcpy(to + offset, data, bytes);
}

static void op_copy(Exact *gpu, int from, uint32_t from_offset, int to, uint32_t to_offset, uint32_t bytes)
{
    /* The texture snapshot (draw: the whole of VRAM into B_SNAPSHOT) copies only the pages written
       since the last one: the front end's texture_dirty, cleared right after this call, holds every
       page a draw, upload or local copy changed since then. A GPU copies 4 MB in no time; a CPU
       would copy it before every draw that samples a fresh render target. */
    if (from == B_VRAM && to == B_SNAPSHOT && !from_offset && !to_offset && bytes == GS_VRAM_SIZE) {
        memcpy(command_add((Cpu *)gpu, CMD_SNAPSHOT)->pages, gpu->texture_dirty, sizeof gpu->texture_dirty);
        return;
    }
    Command *c = command_add((Cpu *)gpu, CMD_COPY);
    c->from = from; c->to = to; c->from_offset = from_offset; c->to_offset = to_offset; c->bytes = bytes;
}

static void op_batch(Exact *gpu, uint32_t offset, uint32_t x, uint32_t y)
{
    (void)x; (void)y;
    command_add((Cpu *)gpu, CMD_BATCH)->offset = offset;
}

static int op_record(Exact *gpu, int pipeline, uint32_t x, uint32_t y)
{
    Cpu *cpu = (Cpu *)gpu;
    (void)x; (void)y;
    if (pipeline == P_COPY) memcpy(command_add(cpu, CMD_LOCAL_COPY)->c, cpu->copy_constants, 48);
    else if (pipeline == P_SCANOUT) memcpy(command_add(cpu, CMD_SCANOUT)->c, cpu->display_constants, 48);
    return 1;
}

static void gpu_submit(Cpu *cpu);
static struct {
    int on;
    GnmKernel raster, copy, local_copy, scanout;
    uint64_t dispatches, waits;
    /* the arenas' GPU copies in garlic memory: the front end keeps its own in cached memory, which
       it reads back as it builds them; a submit copies the used parts over (sequential writes) */
    Primitive *primitives;
    Constants *states;
    uint32_t *bins, *batches;
    /* experiments, from files in /data/dhd: gpu_no_raster (no batch dispatches), gpu_no_barrier
       (no barriers: wrong pictures, for timing only), gpu_no_pixels (the raster kernel's mode 1) */
    int no_raster, no_barrier, no_pixels;
    /* the profile (gpu_submit, one submit in every 10 s line): GPU time per kind, the slowest batches */
    int profile_next;
    uint64_t kind_ticks[5], kind_count[5];
    struct { uint64_t ticks; uint32_t tiles, members, primitives, types[8]; } slow[4];
} device;

/* the profiled submit's dispatches: kind, tiles, list members, primitive types (GPU thread only) */
static struct { int kind; uint32_t tiles, members, types[8]; } profile_marks[4000];

static void op_submit(Exact *gpu)
{
    Cpu *cpu = (Cpu *)gpu;
    if (device.on) { gpu_submit(cpu); return; }
    uint64_t start = sceKernelGetProcessTime();
    for (unsigned i = 0; i < cpu->command_count; i++) {
        const Command *c = &cpu->commands[i];
        switch (c->kind) {
        case CMD_COPY:
            memmove(buffer_base(cpu, c->to) + c->to_offset, buffer_base(cpu, c->from) + c->from_offset, c->bytes);
            break;
        case CMD_SNAPSHOT:
            cpu->snapshots++;
            for (unsigned page = 0; page < 512;) {
                if (!(c->pages[page / 32] >> (page & 31) & 1)) { page++; continue; }
                unsigned end = page;
                while (end < 512 && c->pages[end / 32] >> (end & 31) & 1) end++;
                memcpy((uint8_t *)cpu->snapshot + page * 8192, (uint8_t *)cpu->vram + page * 8192, (end - page) * 8192);
                cpu->snapshot_pages += end - page;
                page = end;
            }
            break;
        case CMD_BATCH: {
            uint64_t key = batch_key(cpu, c->offset);
            unsigned count = 1;
            if (key)
                while (i + count < cpu->command_count && cpu->commands[i + count].kind == CMD_BATCH &&
                       batch_key(cpu, cpu->commands[i + count].offset) == key)
                    count++;
            if (count > 1) { run_batches(cpu, c, count); i += count - 1; }
            else run_batch(cpu, c->offset);
            break;
        }
        case CMD_LOCAL_COPY: run_local_copy(cpu, c->c); break;
        case CMD_SCANOUT: run_scanout(cpu, c->c, c->c[5], c->c[6]); break;
        }
    }
    cpu->command_count = 0;
    cpu->raster_ticks += sceKernelGetProcessTime() - start;
}


/* ---- the GPU device (port/ps4/gnm.c, port/ps4/kernels/raster.cl) ----

   The same commands as the CPU's, each one dispatch of raster.cl's kernels, with a barrier after
   each (a batch draws over the last one's pixels; WebGPU's passes order their dispatches the same
   way); a submit is one command buffer, waited for. VRAM, the texture snapshot and the upload pages
   are garlic memory (the GPU's fast path; the CPU only writes the upload pages); what the CPU reads
   (the readback and staging pages, the scanned lines, the front end's arenas) is onion memory. */

static void gpu_copy(Cpu *cpu, const void *from, void *to, uint32_t bytes)
{
    (void)cpu;
    struct { uint64_t from, to; uint32_t words, stride; } a;
    uint32_t words = bytes / 4, groups = (words + 63) / 64;
    if (groups > 1024) groups = 1024;
    a.from = (uint64_t)(uintptr_t)from; a.to = (uint64_t)(uintptr_t)to; a.words = words; a.stride = groups * 64u;
    gnm_dispatch(&device.copy, &a, sizeof a, groups, 1, 64, 1);
}

static void gpu_submit(Cpu *cpu)
{
    uint64_t start = sceKernelGetProcessTime();
    const uint32_t *scanout = NULL;
    Exact *e = &cpu->e;
    int profile = device.profile_next && cpu->command_count >= 20 && cpu->command_count < 4000;
    if (profile) device.profile_next = 0;
    memset(profile_marks, 0, sizeof profile_marks);
    memcpy(device.primitives, e->primitive_data, e->primitive_count * sizeof(Primitive));
    memcpy(device.states, e->state_data, e->state_count * sizeof(Constants));
    memcpy(device.bins, e->bin_data, (size_t)e->bin_words * 4);
    memcpy(device.batches, e->batch_data, (size_t)e->batch_count * BATCH_STRIDE);
    gnm_begin();
    unsigned dispatched = 0;
    for (unsigned i = 0; i < cpu->command_count; i++) {
        const Command *c = &cpu->commands[i];
        if (!gnm_room()) {  /* a full buffer goes now; the rest follow in another */
            profile = 0;
            if (!gnm_finish()) break;
            gnm_begin();
        }
        if (profile && dispatched == 0) gnm_mark(0);
        uint32_t profile_tiles = 0, profile_members = 0;
        switch (c->kind) {
        case CMD_COPY:
            gpu_copy(cpu, buffer_base(cpu, c->from) + c->from_offset, buffer_base(cpu, c->to) + c->to_offset, c->bytes);
            break;
        case CMD_SNAPSHOT:
            cpu->snapshots++;
            for (unsigned page = 0; page < 512;) {
                if (!(c->pages[page / 32] >> (page & 31) & 1)) { page++; continue; }
                unsigned end = page;
                while (end < 512 && c->pages[end / 32] >> (end & 31) & 1) end++;
                gpu_copy(cpu, (uint8_t *)cpu->vram + page * 8192, (uint8_t *)cpu->snapshot + page * 8192, (end - page) * 8192);
                cpu->snapshot_pages += end - page;
                page = end;
            }
            break;
        case CMD_BATCH: {
            const uint32_t *header = cpu->e.batch_data + c->offset / 4;
            uint32_t touched = header[0];
            if (!touched) continue;
            if (device.no_raster) continue;
            struct { uint64_t bins, primitives, states, vram, snapshot; uint32_t touched, base, columns, mode; } a;
            uint32_t columns = touched < 32768u ? touched : 32768u;
            a.bins = (uint64_t)(uintptr_t)device.bins;
            a.primitives = (uint64_t)(uintptr_t)device.primitives;
            a.states = (uint64_t)(uintptr_t)device.states;
            a.vram = (uint64_t)(uintptr_t)cpu->vram;
            a.snapshot = (uint64_t)(uintptr_t)cpu->snapshot;
            a.touched = touched; a.base = header[1]; a.columns = columns; a.mode = device.no_pixels ? 1u : 0u;
            gnm_dispatch(&device.raster, &a, sizeof a, columns, (touched + columns - 1) / columns, 8, 8);
            if (profile) {
                profile_tiles = touched;
                for (uint32_t t = 0; t < touched; t++) {
                    uint32_t first = e->bin_data[header[1] + t * 3u + 1], count = e->bin_data[header[1] + t * 3u + 2];
                    profile_members += count;
                    for (uint32_t q = 0; q < count; q++)
                        profile_marks[dispatched].types[e->primitive_data[e->bin_data[first + q]].info[0] & 7u]++;
                }
            }
            cpu->tiles += touched;
            cpu->batches++;
            break;
        }
        case CMD_LOCAL_COPY: {
            struct { uint64_t vram; uint32_t c[12]; } a;
            a.vram = (uint64_t)(uintptr_t)cpu->vram;
            memcpy(a.c, c->c, sizeof a.c);
            gnm_dispatch(&device.local_copy, &a, sizeof a, (c->c[8] + 7) / 8, (c->c[9] + 7) / 8, 8, 8);
            break;
        }
        case CMD_SCANOUT: {
            struct { uint64_t vram, out; uint32_t c[12]; uint32_t lines; } a;
            a.vram = (uint64_t)(uintptr_t)cpu->vram;
            a.out = (uint64_t)(uintptr_t)cpu->field;
            memcpy(a.c, c->c, sizeof a.c);
            a.lines = scan_lines(c->c);
            gnm_dispatch(&device.scanout, &a, sizeof a, (c->c[5] + 7) / 8, (a.lines + 7) / 8, 8, 8);
            scanout = c->c;
            break;
        }
        }
        if (!device.no_barrier) gnm_barrier();
        if (profile) {
            profile_marks[dispatched].kind = c->kind;
            profile_marks[dispatched].tiles = profile_tiles;
            profile_marks[dispatched].members = profile_members;
            gnm_mark(dispatched + 1);
        }
        dispatched++;
    }
    int ok = gnm_finish();
    device.dispatches += dispatched;
    device.waits++;
    if (!ok) {
        /* the GPU stopped answering (gnm.c logged it): the CPU draws from here on */
        fprintf(stderr, "gs: the GPU device failed; drawing on the CPU from now on\n");
        device.on = 0;
        op_submit(&cpu->e);
        return;
    }
    if (profile && dispatched) {
        for (unsigned k = 0; k < dispatched; k++) {
            uint64_t t = gnm_mark_time(k + 1) - gnm_mark_time(k);
            int kind = profile_marks[k].kind;
            device.kind_ticks[kind] += t;
            device.kind_count[kind]++;
            if (kind != CMD_BATCH) continue;
            for (int r = 0; r < 4; r++)
                if (t > device.slow[r].ticks) {
                    for (int q = 3; q > r; q--) device.slow[q] = device.slow[q - 1];
                    device.slow[r].ticks = t;
                    device.slow[r].tiles = profile_marks[k].tiles;
                    device.slow[r].members = profile_marks[k].members;
                    memcpy(device.slow[r].types, profile_marks[k].types, sizeof device.slow[r].types);
                    break;
                }
        }
        static const char *names[5] = {"copy", "snapshot", "batch", "local copy", "scanout"};
        char line[1024];
        int n = snprintf(line, sizeof line, "gs gpu profile: one submit of %u dispatches:", dispatched);
        for (int kind = 0; kind < 5; kind++)
            if (device.kind_count[kind])
                n += snprintf(line + n, sizeof line - n, " %s %u x %.2f ms", names[kind], (unsigned)device.kind_count[kind],
                              device.kind_ticks[kind] / 1e5);
        n += snprintf(line + n, sizeof line - n, "; slowest batches:");
        for (int r = 0; r < 4 && device.slow[r].ticks; r++)
            n += snprintf(line + n, sizeof line - n, " %.2f ms (%u tiles, %u in lists; types 0-6: %u %u %u %u %u %u %u)",
                          device.slow[r].ticks / 1e5, device.slow[r].tiles, device.slow[r].members,
                          device.slow[r].types[0], device.slow[r].types[1], device.slow[r].types[2], device.slow[r].types[3],
                          device.slow[r].types[4], device.slow[r].types[5], device.slow[r].types[6]);
        fprintf(stderr, "%s\n", line);
        memset(device.kind_ticks, 0, sizeof device.kind_ticks);
        memset(device.kind_count, 0, sizeof device.kind_count);
        memset(device.slow, 0, sizeof device.slow);
    }
    if (scanout) compose_scanout(cpu, scanout, scanout[5], scanout[6]);
    cpu->command_count = 0;
    cpu->raster_ticks += sceKernelGetProcessTime() - start;
}

/* The device's memory and kernels; 0 leaves the CPU device. Its arenas replace exact_alloc's. */
static int gpu_device_start(Cpu *cpu)
{
    if (!gnm_init()) return 0;
    if (!gnm_kernel("raster", &device.raster) || !gnm_kernel("copy_words", &device.copy) ||
        !gnm_kernel("local_copy", &device.local_copy) || !gnm_kernel("scanout", &device.scanout)) {
        fprintf(stderr, "gs: the GPU kernels are missing\n");
        return 0;
    }
    uint32_t *vram = gnm_alloc_garlic(GS_VRAM_SIZE, 65536), *snapshot = gnm_alloc_garlic(GS_VRAM_SIZE, 65536);
    uint32_t *upload = gnm_alloc_garlic(GS_VRAM_SIZE, 65536);
    uint32_t *readback = gnm_alloc(GS_VRAM_SIZE, 65536), *staged = gnm_alloc(GS_VRAM_SIZE, 65536);
    uint32_t *screen = gnm_alloc(1024 * 1024 * 4, 65536), *field = gnm_alloc(1024 * 1024 * 4, 65536);
    Primitive *primitives = gnm_alloc_garlic(MAX_PRIMITIVES * sizeof(Primitive), 256);
    Constants *states = gnm_alloc_garlic(MAX_STATES * sizeof(Constants), 256);
    uint32_t *bins = gnm_alloc_garlic(MAX_BIN_WORDS * 4, 256), *batches = gnm_alloc_garlic(MAX_BATCHES * BATCH_STRIDE, 256);
    if (!vram || !snapshot || !upload || !readback || !staged || !screen || !field || !primitives || !states ||
        !bins || !batches) {
        fprintf(stderr, "gs: not enough GPU memory for the GPU device\n");
        return 0;
    }
    memset(vram, 0, GS_VRAM_SIZE); memset(snapshot, 0, GS_VRAM_SIZE); memset(upload, 0, GS_VRAM_SIZE);
    memset(readback, 0, GS_VRAM_SIZE); memset(staged, 0, GS_VRAM_SIZE);
    memset(screen, 0, 1024 * 1024 * 4); memset(field, 0, 1024 * 1024 * 4);
    memset(batches, 0, MAX_BATCHES * BATCH_STRIDE);
    device.primitives = primitives; device.states = states; device.bins = bins; device.batches = batches;
    cpu->vram = vram; cpu->snapshot = snapshot; cpu->upload = upload; cpu->readback = readback; cpu->staged = staged;
    cpu->screen = screen; cpu->field = field;
    fprintf(stderr, "gs: GPU renderer (raster %u VGPRs; arguments %u/%u bytes)\n",
            ((device.raster.rsrc1 & 63u) + 1u) * 4u, (unsigned)sizeof(struct { uint64_t p[5]; uint32_t u[3]; }),
            device.raster.kernarg_size);
    struct stat st;
    device.no_raster = stat("/data/dhd/gpu_no_raster", &st) == 0;
    device.no_barrier = stat("/data/dhd/gpu_no_barrier", &st) == 0;
    device.no_pixels = stat("/data/dhd/gpu_no_pixels", &st) == 0;
    if (device.no_raster || device.no_barrier || device.no_pixels)
        fprintf(stderr, "gs: GPU experiment:%s%s%s\n", device.no_raster ? " no raster" : "",
                device.no_barrier ? " no barriers" : "", device.no_pixels ? " no pixels" : "");
    device.on = 1;
    return 1;
}

static void op_read(Exact *gpu, int buffer, const uint32_t want[16], void *image)
{
    const uint8_t *from = buffer_base((Cpu *)gpu, buffer);
    for (unsigned page = 0; page < 512; page++)
        if (want[page / 32] >> (page & 31) & 1)
            memcpy((uint8_t *)image + (size_t)page * 8192, from + (size_t)page * 8192, 8192);
}

static uint64_t op_clock(void) { return sceKernelGetProcessTime(); }
static int op_clip(const Exact *gpu, int box[4]) { (void)gpu; (void)box; return 0; }
static int op_draw_skipped(const Gs *g) { (void)g; return 0; }

/* ---- the picture on the screen: sceVideoOut ---- */

#define OUT_WIDTH 1920
#define OUT_HEIGHT 1080
#define OUT_BUFFERS 3
#define OUT_BUFFER_SIZE (OUT_WIDTH * OUT_HEIGHT * 4)
#define OUT_ALIGN 0x200000

static struct {
    int handle;
    OrbisKernelEqueue queue;
    uint32_t *buffers[OUT_BUFFERS];
    int current;
    int64_t flips;
    unsigned left[OUT_BUFFERS], width[OUT_BUFFERS];  /* the picture's columns each buffer last had */
    unsigned short columns[OUT_WIDTH], rows[OUT_HEIGHT];
    unsigned map_source_width, map_source_height, map_width;
} out = {.handle = -1};

static int video_open(void)
{
    if (out.handle >= 0) return 1;
    int handle = sceVideoOutOpen(ORBIS_VIDEO_USER_MAIN, ORBIS_VIDEO_OUT_BUS_MAIN, 0, NULL);
    if (handle < 0) { fprintf(stderr, "gs: sceVideoOutOpen failed (0x%08x)\n", (unsigned)handle); return 0; }
    if (sceKernelCreateEqueue(&out.queue, "dhd flip") != 0) { fprintf(stderr, "gs: no flip queue\n"); return 0; }
    sceVideoOutAddFlipEvent(out.queue, handle, NULL);
    size_t size = (OUT_BUFFER_SIZE * OUT_BUFFERS + OUT_ALIGN - 1) / OUT_ALIGN * OUT_ALIGN;
    off_t offset;
    void *memory = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, OUT_ALIGN, 3, &offset) != 0) {
        fprintf(stderr, "gs: no direct memory for the display buffers\n");
        return 0;
    }
    if (sceKernelMapDirectMemory(&memory, size, 0x33, 0, offset, OUT_ALIGN) != 0) {
        fprintf(stderr, "gs: the display buffers could not be mapped\n");
        sceKernelReleaseDirectMemory(offset, size);
        return 0;
    }
    for (int i = 0; i < OUT_BUFFERS; i++) {
        out.buffers[i] = (uint32_t *)((char *)memory + (size_t)i * OUT_BUFFER_SIZE);
        memset(out.buffers[i], 0, OUT_BUFFER_SIZE);
    }
    OrbisVideoOutBufferAttribute attribute;
    memset(&attribute, 0, sizeof attribute);
    sceVideoOutSetBufferAttribute(&attribute, 0x80000000, 1, 0, OUT_WIDTH, OUT_HEIGHT, OUT_WIDTH);
    if (sceVideoOutRegisterBuffers(handle, 0, (void **)out.buffers, OUT_BUFFERS, &attribute) != 0) {
        fprintf(stderr, "gs: sceVideoOutRegisterBuffers failed\n");
        return 0;
    }
    sceVideoOutSetFlipRate(handle, 0);
    out.handle = handle;
    sceSystemServiceHideSplashScreen();
    fprintf(stderr, "gs: video out %dx%d, %d buffers\n", OUT_WIDTH, OUT_HEIGHT, OUT_BUFFERS);
    return 1;
}

static void wait_flip(int64_t flip)
{
    if (flip <= 0) return;
    for (;;) {
        OrbisVideoOutFlipStatus status;
        if (sceVideoOutGetFlipStatus(out.handle, &status) != 0 || (int64_t)status.flipArg >= flip) return;
        OrbisKernelEvent event;
        int count;
        if (sceKernelWaitEqueue(out.queue, &event, 1, &count, NULL) != 0) return;
    }
}

/* pixels (RGBA, width x height) scaled into the next display buffer (nearest; the shape of the
   widescreen setting, else 4:3, as tall as the screen), then flipped at the next vertical blank. */
static void video_show(const uint32_t *pixels, unsigned width, unsigned height)
{
    wait_flip(out.flips - 1);  /* at most one flip in flight: the buffer filled now is not shown */
    double aspect = host_widescreen_aspect();
    if (aspect <= 0) aspect = 4.0 / 3.0;
    unsigned picture_width = (unsigned)(OUT_HEIGHT * aspect + 0.5);
    if (picture_width > OUT_WIDTH) picture_width = OUT_WIDTH;
    picture_width &= ~1u;
    unsigned left = (OUT_WIDTH - picture_width) / 2;
    if (out.map_source_width != width || out.map_source_height != height || out.map_width != picture_width) {
        for (unsigned x = 0; x < picture_width; x++) out.columns[x] = (unsigned short)(x * width / picture_width);
        for (unsigned y = 0; y < OUT_HEIGHT; y++) out.rows[y] = (unsigned short)(y * height / OUT_HEIGHT);
        out.map_source_width = width; out.map_source_height = height; out.map_width = picture_width;
    }
    uint32_t *buffer = out.buffers[out.current];
    if (out.left[out.current] != left || out.width[out.current] != picture_width) {
        memset(buffer, 0, OUT_BUFFER_SIZE);  /* the bars beside the picture */
        out.left[out.current] = left; out.width[out.current] = picture_width;
    }
    static uint32_t row[OUT_WIDTH];
    int converted = -1;
    for (unsigned y = 0; y < OUT_HEIGHT; y++) {
        if (out.rows[y] != converted) {
            converted = out.rows[y];
            const uint32_t *source = pixels + (size_t)converted * width;
            for (unsigned x = 0; x < picture_width; x++) {
                uint32_t v = source[out.columns[x]];
                row[x] = 0xFF000000u | ((v & 255u) << 16) | (v & 0xFF00u) | ((v >> 16) & 255u);
            }
        }
        memcpy(buffer + (size_t)y * OUT_WIDTH + left, row, picture_width * 4);
    }
    out.flips++;
    sceVideoOutSubmitFlip(out.handle, out.current, ORBIS_VIDEO_OUT_FLIP_VSYNC, out.flips);
    out.current = (out.current + 1) % OUT_BUFFERS;
}

/* ---- the renderer's interface (gpu.h) ---- */

static unsigned render_absence;

int gs_gpu_init(Gs *g)
{
    Cpu *cpu = calloc(1, sizeof *cpu);
    if (!cpu) return 0;
    g->backend = cpu;
    if (!exact_alloc(&cpu->e)) {
        fprintf(stderr, "gs: out of memory for the renderer\n");
        gs_gpu_destroy(g);
        return 0;
    }
    if (!gpu_device_start(cpu)) {
        cpu->vram = calloc(1, GS_VRAM_SIZE);
        cpu->snapshot = calloc(1, GS_VRAM_SIZE);
        cpu->readback = calloc(1, GS_VRAM_SIZE);
        cpu->staged = calloc(1, GS_VRAM_SIZE);
        cpu->upload = calloc(1, GS_VRAM_SIZE);
        cpu->screen = calloc(1024 * 1024, 4);
        cpu->field = calloc(1024 * 1024, 4);
    }
    if (!cpu->vram || !cpu->snapshot || !cpu->readback || !cpu->staged || !cpu->upload || !cpu->screen || !cpu->field) {
        fprintf(stderr, "gs: out of memory for the CPU renderer\n");
        gs_gpu_destroy(g);
        return 0;
    }
    const char *limit = getenv("DHD_GS_BATCH");
    exact_config(&cpu->e, getenv("DHD_PROFILE") != NULL, limit ? strtoul(limit, NULL, 0) : 0);
    exact_hooks(g, &cpu->e);
    if (!device.on && !pool.workers) pool_start();
    return 1;
}

void gs_gpu_destroy(Gs *g)
{
    Cpu *cpu = g->backend;
    if (!cpu) return;
    if (cpu->e.primitive_data) {
        sync_vram(g);
        if (cpu->e.recorded) submit_all(&cpu->e);
    }
    if (!device.on) {  /* the GPU device's buffers are the pool's, kept for the next renderer */
        free(cpu->vram); free(cpu->snapshot); free(cpu->readback); free(cpu->staged); free(cpu->upload);
        free(cpu->screen); free(cpu->field);
    }
    exact_free(&cpu->e);
    free(cpu->commands);
    free(cpu);
    g->backend = NULL; g->draw = NULL; g->sync = NULL; g->sync_range = NULL;
    g->local_copy = NULL; g->stage_pages = NULL; g->fetch_pages = NULL;
}

void gs_gpu_profile(Gs *g, double seconds)
{
    Cpu *cpu = g->backend;
    if (!cpu || !cpu->e.profile) return;
    Exact *e = &cpu->e;
    fprintf(stderr, "profile GS (cpu): %.1f draws/s, %.1f batches/s, %.0f tiles/s, raster %.1f ms/s, "
            "%.1f snapshots/s, %.1f uploads/s, %.1f submits/s, %.1f local copies/s\n",
            e->draws / seconds, cpu->batches / seconds, cpu->tiles / seconds, cpu->raster_ticks / 1000.0 / seconds,
            e->snapshots / seconds, e->uploads / seconds, e->submits / seconds, e->copies / seconds);
    e->draw_ticks = e->sync_ticks = e->draws = e->snapshots = e->uploads = 0;
    e->dispatches = e->submits = e->copies = 0;
    cpu->batches = cpu->tiles = cpu->raster_ticks = 0;
}

/* The scanout, and a log line when the display circuit's picture moves or changes size (the
   first 40 changes), and every 10 s the CPU renderer's own counts. */
int gs_gpu_display(Gs *g, unsigned *width, unsigned *height)
{
    Cpu *cpu = g->backend;
    int shown = exact_display(g, width, height);
    if (cpu && shown) {
        static uint32_t last[8];
        static unsigned changes;
        if (memcmp(last, cpu->display_constants, sizeof last) && changes < 40) {
            const uint32_t *c = cpu->display_constants;
            changes++;
            fprintf(stderr, "gs: display psm %u fbp %u fbw %u at %u,%u size %ux%u field mode %u\n",
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
            memcpy(last, cpu->display_constants, sizeof last);
        }
    }
    static uint64_t since;
    static unsigned fields;
    uint64_t now = sceKernelGetProcessTime();
    fields++;
    if (cpu && !since) since = now;
    else if (cpu && now - since >= 10000000) {
        double seconds = (now - since) / 1e6;
        fprintf(stderr, "gs %s: %.1f fields/s, %.0f draws/s, %.0f batches/s, %.0f tiles/s, raster %.0f ms/s, "
                "%.1f snapshots/s (%.0f pages/s), %.1f uploads/s, %.1f local copies/s, %.1f submits/s, "
                "GPU wait %.0f ms/s (GPU clock %.0f Mticks/s), %.1f readbacks/s, %.0f dispatches/s\n",
                device.on ? "gpu" : "cpu", fields / seconds, cpu->e.draws / seconds, cpu->batches / seconds, cpu->tiles / seconds,
                cpu->raster_ticks / 1000.0 / seconds, cpu->snapshots / seconds, cpu->snapshot_pages / seconds,
                cpu->e.uploads / seconds, cpu->e.copies / seconds, gnm_submits / seconds, gnm_wait_us / 1000.0 / seconds,
                gnm_gpu_ticks / 1e6 / seconds, cpu->e.reads / seconds, device.dispatches / seconds);
        gnm_submits = gnm_wait_us = gnm_gpu_ticks = 0;
        device.profile_next = 1;
        device.dispatches = 0;
        cpu->e.reads = 0;
        cpu->e.draws = cpu->e.uploads = cpu->e.copies = 0;
        cpu->batches = cpu->tiles = cpu->raster_ticks = cpu->snapshots = cpu->snapshot_pages = 0;
        since = now; fields = 0;
    }
    return shown;
}

int gs_gpu_capture(Gs *g, uint32_t *pixels)
{
    Cpu *cpu = g->backend;
    if (!cpu || !cpu->e.display_valid) return 0;
    submit_all(&cpu->e);
    memcpy(pixels, cpu->screen, (size_t)cpu->screen_width * cpu->screen_height * 4);
    return 1;
}

const char *gs_gpu_backend_name(const Gs *g) { return g && g->backend ? "cpu" : "none"; }
int gs_gpu_surface(Gs *g, const GsWindow *window) { (void)window; return g->backend && video_open(); }
int gs_gpu_surface_reset(Gs *g, const GsWindow *window) { (void)window; return g->backend && video_open(); }

int gs_gpu_present(Gs *g, const uint32_t *pixels, unsigned width, unsigned height,
                   unsigned window_width, unsigned window_height)
{
    (void)window_width; (void)window_height;
    Cpu *cpu = g->backend;
    if (!cpu || out.handle < 0) return 0;
    submit_all(&cpu->e);  /* the scanout */
    if (pixels) {
        if (!width || !height || width > 1024 || height > 1024) return 0;
        memcpy(cpu->screen, pixels, (size_t)width * height * 4);
        cpu->screen_width = width; cpu->screen_height = height;
        cpu->e.display_valid = 1;
    }
    if (!cpu->e.display_valid || !cpu->screen_width || !cpu->screen_height) return 1;
    video_show(cpu->screen, cpu->screen_width, cpu->screen_height);
    gnm_frame_done();
    perf_presented();
    return 1;
}

unsigned gs_render_leave(void) { return ++render_absence; }
unsigned gs_render_generation(void) { return render_absence; }
unsigned gs_render_current(void) { return render_absence; }
int gs_render_release(unsigned generation) { return generation == render_absence; }
int gs_render_quiet(void) { return 1; }
int gs_gpu_lost(void) { return 0; }

/* No native renderer on the PS4 (DHD_RENDER=native needs the GPU): its hooks do nothing. */
int gs_gpu_native(const Gs *g) { (void)g; return 0; }
void gs_native_vsync(Gs *g, unsigned *width, unsigned *height, int resize)
{
    (void)g; (void)resize; *width = *height = 0;
}
void gs_native_vsync_handed(Gs *g, unsigned *width, unsigned *height, int resize, double due, double began,
                            int flat_limit, int card_2d, double output_aspect)
{
    (void)g; (void)resize; (void)due; (void)began; (void)flat_limit; (void)card_2d; (void)output_aspect;
    *width = *height = 0;
}
void gs_native_bench(Gs *g, int end) { (void)g; (void)end; }
void gs_native_lock(Gs *g, int lock) { (void)g; (void)lock; }
void gs_native_present_options(Gs *g) { (void)g; }
uint32_t *gs_native_picture(Gs *g, unsigned *width, unsigned *height) { (void)g; *width = *height = 0; return NULL; }
void gs_native_movie(Gs *g, const uint32_t *rgba, unsigned width, unsigned height, unsigned rows)
{
    (void)g; (void)rgba; (void)width; (void)height; (void)rows;
}

/* gpu_rust.c's and native_core.inc's other calls */
void gs_gpu_counter_apply(void) {}
int gs_native_card(void) { return 0; }  /* the native renderer's card mode: none */
/* The renderer's scale (1x on the CPU device); 2 asks whether a software adapter draws: none. */
int host_service_gs_scale(int asked) { return asked == 2 ? 0 : 1; }
