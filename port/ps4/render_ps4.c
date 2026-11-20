/* The renderer library on the PS4 (port/ps4/PLAN.md): port/render/include/dhd_render.h in C over the
   console's GPU (port/ps4/gnm.c), in place of the Rust/wgpu library, which has no PS4 backend. The
   C side is every platform's: port/gs/gpu_rust.c, gpu_rust_shared.inc and native_core.inc call these
   functions as they call Rust's.

   - Exact (dhdr_exact_*): raster.wgsl's entry points as the GCN kernels of port/ps4/kernels/raster.cl
     (raster_core.h's arithmetic). The 11 buffers are GPU memory: those the GPU works in (VRAM, the
     texture snapshot, the upload pages, the states, primitives, tile lists and batch records)
     garlic, those the CPU reads back (readback, staging, the scanned picture) onion. Writes are
     copies made at once; commands wait in a list and go to the GPU at dhdr_exact_submit as one
     command buffer, a barrier after each dispatch (a batch draws over the last one's pixels, as a
     WebGPU pass orders its dispatches), and the submit waits for them.
   - The display texture: the scanout kernel's lines, made into the picture on the CPU (a field's
     lines doubled and aligned with the picture before; see compose).
   - Surfaces and presentation (dhdr_surface_*, dhdr_present_exact): sceVideoOut, three 1920x1080
     buffers; the picture scaled to the screen's height in the widescreen setting's shape (else 4:3).
   - Native (dhdr_native_*): native.wgsl's renderer on the GPU's graphics pipeline (below; the
     stages are port/ps4/kernels/graphics/native.ll and native.cl, its compute work native_ops.cl). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <orbis/SystemService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include "../render/include/dhd_render.h"
#include "../render/include/dhd_render_ext.h"
#include "gnm.h"

double host_widescreen_aspect(void);  /* port/hle/widescreen.c: the picture's shape, 0 for 4:3 */

/* ---- the device ---- */

static int device_made;

int dhdr_device_init(const DhdrDeviceDesc *d)
{
    (void)d;
    if (!device_made) device_made = gnm_init() ? 1 : -1;
    return device_made > 0;
}
const char *dhdr_adapter_name(void) { return "PS4 GPU (AMD GCN 1.1, PM4 and LLVM kernels)"; }
uint32_t dhdr_device_features(void) { return 0; }
const char *dhdr_version(void) { return "dhd_render ps4 (C over GNM)"; }
/* 4096 texels a side: native_scale_hold then allows scale 2 (4096 / 2048), what the PS4 draws well */
uint32_t dhdr_device_max_texture(void) { return 4096; }
int dhdr_device_lost(void) { return 0; }
uint32_t dhdr_perf_read(uint64_t *out, uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) out[i] = i == 0 ? 2 : 0;  /* [0]: no GPU timer here */
    return count;
}

/* ---- exact ---- */

enum { CMD_COPY, CMD_BATCH, CMD_LOCAL_COPY, CMD_SCANOUT };
typedef struct {
    int kind;
    uint32_t from, to, from_offset, to_offset, bytes;  /* CMD_COPY */
    uint32_t offset;                                   /* CMD_BATCH: the batch record */
    uint32_t c[12], width, height;                     /* CMD_LOCAL_COPY, CMD_SCANOUT */
} Command;

static const uint32_t buffer_size[DHDR_B_COUNT] = {
    [DHDR_B_VRAM] = 4u << 20, [DHDR_B_SNAPSHOT] = 4u << 20, [DHDR_B_STATES] = 4u << 20,
    [DHDR_B_PRIMITIVES] = 4u << 20, [DHDR_B_BINS] = 16u << 20, [DHDR_B_DISPLAY] = 256, [DHDR_B_COPY] = 256,
    [DHDR_B_READBACK] = 4u << 20, [DHDR_B_STAGED] = 4u << 20, [DHDR_B_BATCHES] = 1u << 20, [DHDR_B_UPLOAD] = 4u << 20,
};

struct DhdrExact {
    uint8_t *buffer[DHDR_B_COUNT];
    Command *commands;
    unsigned command_count, command_capacity;
    uint32_t *field;                 /* the scanout kernel's lines (onion) */
    uint32_t *display;               /* the display texture: RGBA, red in the low byte */
    unsigned display_width, display_height;
    GnmKernel raster, copy, local_copy, scanout;
    DhdrExactStats stats;
};

static GnmKernel kernel_or_fail(const char *name, int *ok)
{
    GnmKernel k;
    memset(&k, 0, sizeof k);
    if (!gnm_kernel(name, &k)) { fprintf(stderr, "dhd_render: no %s kernel\n", name); *ok = 0; }
    return k;
}

DhdrExact *dhdr_exact_create(uint32_t raster_flags)
{
    (void)raster_flags;  /* the kernels are raster.wgsl with its options on (ADDRESS_SHIFT, EARLY_DEPTH) */
    if (device_made <= 0) return NULL;
    DhdrExact *x = calloc(1, sizeof *x);
    if (!x) return NULL;
    int ok = 1;
    x->raster = kernel_or_fail("raster", &ok);
    x->copy = kernel_or_fail("copy_words", &ok);
    x->local_copy = kernel_or_fail("local_copy", &ok);
    x->scanout = kernel_or_fail("scanout", &ok);
    for (int b = 0; b < DHDR_B_COUNT && ok; b++) {
        int cpu_reads = b == DHDR_B_READBACK || b == DHDR_B_STAGED || b == DHDR_B_DISPLAY || b == DHDR_B_COPY;
        x->buffer[b] = cpu_reads ? gnm_alloc(buffer_size[b], 65536) : gnm_alloc_garlic(buffer_size[b], 65536);
        if (!x->buffer[b]) { fprintf(stderr, "dhd_render: no GPU memory for exact buffer %d\n", b); ok = 0; break; }
        memset(x->buffer[b], 0, buffer_size[b]);
    }
    x->field = ok ? gnm_alloc(1024 * 1024 * 4, 65536) : NULL;
    x->display = ok ? gnm_alloc(1024 * 1024 * 4, 65536) : NULL;
    if (!ok || !x->field || !x->display) {
        free(x);  /* the pool's memory stays spent: once, at start-up */
        return NULL;
    }
    fprintf(stderr, "dhd_render: exact renderer on the GPU (raster %u VGPRs)\n", ((x->raster.rsrc1 & 63u) + 1u) * 4u);
    return x;
}

void dhdr_exact_destroy(DhdrExact *x)
{
    if (!x) return;
    dhdr_exact_submit(x);
    free(x->commands);
    free(x);  /* its GPU memory is the pool's */
}

static Command *command_add(DhdrExact *x, int kind)
{
    if (x->command_count == x->command_capacity) {
        unsigned capacity = x->command_capacity ? x->command_capacity * 2 : 1024;
        Command *grown = realloc(x->commands, capacity * sizeof *grown);
        if (!grown) { fprintf(stderr, "dhd_render: out of memory for commands\n"); abort(); }
        x->commands = grown;
        x->command_capacity = capacity;
    }
    Command *c = &x->commands[x->command_count++];
    memset(c, 0, sizeof *c);
    c->kind = kind;
    return c;
}

void dhdr_exact_write(DhdrExact *x, uint32_t buffer, uint32_t offset, const void *data, uint32_t bytes)
{
    if (buffer >= DHDR_B_COUNT || (uint64_t)offset + bytes > buffer_size[buffer]) {
        fprintf(stderr, "dhd_render: write past buffer %u (%u + %u)\n", buffer, offset, bytes);
        abort();
    }
    memcpy(x->buffer[buffer] + offset, data, bytes);
    x->stats.bytes_written += bytes;
}

void dhdr_exact_copy(DhdrExact *x, uint32_t from, uint32_t from_offset, uint32_t to, uint32_t to_offset, uint32_t bytes)
{
    Command *c = command_add(x, CMD_COPY);
    c->from = from; c->to = to; c->from_offset = from_offset; c->to_offset = to_offset; c->bytes = bytes;
    x->stats.copies++;
}

void dhdr_exact_batch(DhdrExact *x, uint32_t dynamic_offset, uint32_t groups_x, uint32_t groups_y)
{
    (void)groups_x; (void)groups_y;  /* the record holds its tile count */
    command_add(x, CMD_BATCH)->offset = dynamic_offset;
}

void dhdr_exact_record(DhdrExact *x, uint32_t pipeline, uint32_t x_word, uint32_t y_word)
{
    if (pipeline == DHDR_P_LOCAL_COPY) {
        memcpy(command_add(x, CMD_LOCAL_COPY)->c, x->buffer[DHDR_B_COPY], 48);
    } else if (pipeline == DHDR_P_SCANOUT) {
        Command *c = command_add(x, CMD_SCANOUT);
        memcpy(c->c, x->buffer[DHDR_B_DISPLAY], 48);
        c->width = x_word >> 16;
        c->height = y_word >> 16;
    }
}

static void gpu_copy(DhdrExact *x, const void *from, void *to, uint32_t bytes)
{
    struct { uint64_t from, to; uint32_t words, stride; } a;
    uint32_t words = bytes / 4, groups = (words + 63) / 64;
    if (groups > 1024) groups = 1024;
    a.from = (uint64_t)(uintptr_t)from; a.to = (uint64_t)(uintptr_t)to; a.words = words; a.stride = groups * 64u;
    gnm_dispatch(&x->copy, &a, sizeof a, groups, 1, 64, 1);
}

/* The scanout's source lines: a field's (c[7]: interlaced, each line shown twice) or the picture's. */
static uint32_t scan_lines(const uint32_t c[12], uint32_t height) { return c[7] ? (height + 1) / 2 : height; }

/* The display texture from the scanned lines. Field mode: the game draws each field half a line
   apart, for a TV's interlace; a progressive screen shows that as the picture jumping a line up and
   down every field. So a field is shown one line lower when that matches the picture before it
   better (sampled differences of the green channel; a tie keeps the last choice). */
static void compose(DhdrExact *x, const uint32_t c[12], uint32_t width, uint32_t height)
{
    static int shift;  /* the last field's: 0, or 1 = shown one line lower */
    uint32_t lines = scan_lines(c, height);
    if (!c[7]) {
        memcpy(x->display, x->field, (size_t)width * height * 4);
    } else {
        if (x->display_width == width && x->display_height == height && height >= 32) {
            uint64_t cost[2] = {0, 0};
            for (int s = 0; s < 2; s++)
                for (uint32_t y = 8; y + 8 < height; y += 4) {
                    uint32_t line = s ? (y - 1) / 2 : y / 2;
                    const uint32_t *f = x->field + (size_t)line * width, *old = x->display + (size_t)y * width;
                    for (uint32_t i = 4; i < width; i += 16) {
                        int g = (int)((f[i] >> 8) & 255u) - (int)((old[i] >> 8) & 255u);
                        cost[s] += (uint64_t)(g < 0 ? -g : g);
                    }
                }
            if (cost[0] != cost[1]) shift = cost[1] < cost[0];
        }
        for (uint32_t y = 0; y < height; y++) {
            uint32_t line = shift ? (y ? (y - 1) / 2 : 0) : y / 2;
            if (line >= lines) line = lines - 1;
            memcpy(x->display + (size_t)y * width, x->field + (size_t)line * width, width * 4);
        }
    }
    x->display_width = width;
    x->display_height = height;
}

void dhdr_exact_submit(DhdrExact *x)
{
    if (!x->command_count) return;
    const Command *scanout = NULL;
    gnm_begin();
    for (unsigned i = 0; i < x->command_count; i++) {
        const Command *c = &x->commands[i];
        if (!gnm_room()) {  /* a full buffer goes now; the rest follow in another */
            if (!gnm_finish()) break;
            gnm_begin();
        }
        switch (c->kind) {
        case CMD_COPY:
            gpu_copy(x, x->buffer[c->from] + c->from_offset, x->buffer[c->to] + c->to_offset, c->bytes);
            break;
        case CMD_BATCH: {
            const uint32_t *header = (const uint32_t *)(x->buffer[DHDR_B_BATCHES] + c->offset);
            uint32_t touched = header[0];
            if (!touched) continue;
            struct { uint64_t bins, primitives, states, vram, snapshot; uint32_t touched, base, columns, mode; } a;
            uint32_t columns = touched < 32768u ? touched : 32768u;
            a.bins = (uint64_t)(uintptr_t)x->buffer[DHDR_B_BINS];
            a.primitives = (uint64_t)(uintptr_t)x->buffer[DHDR_B_PRIMITIVES];
            a.states = (uint64_t)(uintptr_t)x->buffer[DHDR_B_STATES];
            a.vram = (uint64_t)(uintptr_t)x->buffer[DHDR_B_VRAM];
            a.snapshot = (uint64_t)(uintptr_t)x->buffer[DHDR_B_SNAPSHOT];
            a.touched = touched; a.base = header[1]; a.columns = columns; a.mode = 0;
            gnm_dispatch(&x->raster, &a, sizeof a, columns, (touched + columns - 1) / columns, 8, 8);
            break;
        }
        case CMD_LOCAL_COPY: {
            struct { uint64_t vram; uint32_t c[12]; } a;
            a.vram = (uint64_t)(uintptr_t)x->buffer[DHDR_B_VRAM];
            memcpy(a.c, c->c, sizeof a.c);
            gnm_dispatch(&x->local_copy, &a, sizeof a, (c->c[8] + 7) / 8, (c->c[9] + 7) / 8, 8, 8);
            break;
        }
        case CMD_SCANOUT: {
            if (!c->width || !c->height || c->width > 1024 || c->height > 1024) break;
            struct { uint64_t vram, out; uint32_t c[12]; uint32_t lines; } a;
            a.vram = (uint64_t)(uintptr_t)x->buffer[DHDR_B_VRAM];
            a.out = (uint64_t)(uintptr_t)x->field;
            memcpy(a.c, c->c, sizeof a.c);
            a.c[5] = c->width;  /* the kernel's row width: the display texture's */
            a.lines = scan_lines(c->c, c->height);
            gnm_dispatch(&x->scanout, &a, sizeof a, (c->width + 7) / 8, (a.lines + 7) / 8, 8, 8);
            scanout = c;
            break;
        }
        }
        gnm_barrier();
        x->stats.dispatches++;
    }
    if (!gnm_finish()) {
        fprintf(stderr, "dhd_render: the GPU stopped answering; the game cannot draw\n");
        abort();
    }
    x->stats.submits++;
    if (scanout) compose(x, scanout->c, scanout->width, scanout->height);
    x->command_count = 0;
}

int dhdr_exact_read_begin(DhdrExact *x, uint32_t buffer, const uint32_t want[16])
{
    (void)buffer; (void)want;
    uint64_t start = sceKernelGetProcessTime();
    dhdr_exact_submit(x);  /* the copies into the buffer, finished */
    x->stats.reads++;
    x->stats.read_ticks += (sceKernelGetProcessTime() - start) * 1000;
    return 1;
}

int dhdr_exact_read_finish(DhdrExact *x, uint32_t buffer, const uint32_t want[16], void *image)
{
    const uint8_t *from = x->buffer[buffer];
    for (unsigned page = 0; page < 512; page++)
        if (want[page / 32] >> (page & 31) & 1)
            memcpy((uint8_t *)image + (size_t)page * 8192, from + (size_t)page * 8192, 8192);
    return 1;
}

int dhdr_exact_capture_begin(DhdrExact *x, uint32_t *w, uint32_t *h)
{
    dhdr_exact_submit(x);
    *w = x->display_width; *h = x->display_height;
    return x->display_width != 0;
}

int dhdr_exact_capture_finish(DhdrExact *x, uint32_t *pixels)
{
    memcpy(pixels, x->display, (size_t)x->display_width * x->display_height * 4);
    return 1;
}

void dhdr_exact_stats(const DhdrExact *x, DhdrExactStats *out) { *out = x->stats; }

/* ---- surfaces and presentation: sceVideoOut ---- */

#define OUT_WIDTH 1920
#define OUT_HEIGHT 1080
#define OUT_BUFFERS 3
#define OUT_BUFFER_SIZE (OUT_WIDTH * OUT_HEIGHT * 4)
#define OUT_ALIGN 0x200000

struct DhdrSurface {
    int handle;
    OrbisKernelEqueue queue;
    uint32_t *buffers[OUT_BUFFERS];
    int current, drawn;              /* drawn: buffers[current] holds a picture to flip */
    int64_t flips;
    unsigned left[OUT_BUFFERS], width[OUT_BUFFERS];  /* the picture's columns each buffer last had */
    unsigned short columns[OUT_WIDTH], rows[OUT_HEIGHT];
    unsigned map_source_width, map_source_height, map_width;
};

DhdrSurface *dhdr_surface_create(const DhdrWindow *w, uint32_t present_flags)
{
    (void)w; (void)present_flags;  /* the PS4 has one screen; flips wait for the vertical blank */
    static DhdrSurface *made;
    if (made) return made;
    DhdrSurface *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    int handle = sceVideoOutOpen(ORBIS_VIDEO_USER_MAIN, ORBIS_VIDEO_OUT_BUS_MAIN, 0, NULL);
    if (handle < 0) { fprintf(stderr, "dhd_render: sceVideoOutOpen failed (0x%08x)\n", (unsigned)handle); free(s); return NULL; }
    if (sceKernelCreateEqueue(&s->queue, "dhd flip") != 0) { fprintf(stderr, "dhd_render: no flip queue\n"); free(s); return NULL; }
    sceVideoOutAddFlipEvent(s->queue, handle, NULL);
    size_t size = (OUT_BUFFER_SIZE * OUT_BUFFERS + OUT_ALIGN - 1) / OUT_ALIGN * OUT_ALIGN;
    off_t offset;
    void *memory = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, OUT_ALIGN, 3, &offset) != 0 ||
        sceKernelMapDirectMemory(&memory, size, 0x33, 0, offset, OUT_ALIGN) != 0) {
        fprintf(stderr, "dhd_render: no memory for the display buffers\n");
        free(s);
        return NULL;
    }
    for (int i = 0; i < OUT_BUFFERS; i++) {
        s->buffers[i] = (uint32_t *)((char *)memory + (size_t)i * OUT_BUFFER_SIZE);
        memset(s->buffers[i], 0, OUT_BUFFER_SIZE);
    }
    OrbisVideoOutBufferAttribute attribute;
    memset(&attribute, 0, sizeof attribute);
    sceVideoOutSetBufferAttribute(&attribute, 0x80000000, 1, 0, OUT_WIDTH, OUT_HEIGHT, OUT_WIDTH);
    if (sceVideoOutRegisterBuffers(handle, 0, (void **)s->buffers, OUT_BUFFERS, &attribute) != 0) {
        fprintf(stderr, "dhd_render: sceVideoOutRegisterBuffers failed\n");
        free(s);
        return NULL;
    }
    sceVideoOutSetFlipRate(handle, 0);
    s->handle = handle;
    sceSystemServiceHideSplashScreen();
    fprintf(stderr, "dhd_render: video out %dx%d, %d buffers\n", OUT_WIDTH, OUT_HEIGHT, OUT_BUFFERS);
    return made = s;
}

void dhdr_surface_destroy(DhdrSurface *s) { (void)s; }  /* the one screen stays */
uint32_t dhdr_surface_usage(const DhdrSurface *s) { (void)s; return 0; }
void dhdr_surface_counter(DhdrSurface *s, int enabled, int server_ms, int race_ms, int riders_ms, double seconds)
{
    (void)s; (void)enabled; (void)server_ms; (void)race_ms; (void)riders_ms; (void)seconds;
}
void dhdr_surface_touch(DhdrSurface *s, const float *items, uint32_t count, float width_mm, float height_mm)
{
    (void)s; (void)items; (void)count; (void)width_mm; (void)height_mm;
}

static void wait_flip(DhdrSurface *s, int64_t flip)
{
    if (flip <= 0) return;
    for (;;) {
        OrbisVideoOutFlipStatus status;
        if (sceVideoOutGetFlipStatus(s->handle, &status) != 0 || (int64_t)status.flipArg >= flip) return;
        OrbisKernelEvent event;
        int count;
        if (sceKernelWaitEqueue(s->queue, &event, 1, &count, NULL) != 0) return;
    }
}

/* pixels (RGBA, width x height) scaled into the next display buffer: nearest, as tall as the
   screen, in the widescreen setting's shape (else 4:3), black bars beside. */
static void draw_picture(DhdrSurface *s, const uint32_t *pixels, unsigned width, unsigned height)
{
    wait_flip(s, s->flips - 1);  /* at most one flip in flight: the buffer filled now is not shown */
    double aspect = host_widescreen_aspect();
    if (aspect <= 0) aspect = 4.0 / 3.0;
    unsigned picture_width = (unsigned)(OUT_HEIGHT * aspect + 0.5);
    if (picture_width > OUT_WIDTH) picture_width = OUT_WIDTH;
    picture_width &= ~1u;
    unsigned left = (OUT_WIDTH - picture_width) / 2;
    if (s->map_source_width != width || s->map_source_height != height || s->map_width != picture_width) {
        for (unsigned x = 0; x < picture_width; x++) s->columns[x] = (unsigned short)(x * width / picture_width);
        for (unsigned y = 0; y < OUT_HEIGHT; y++) s->rows[y] = (unsigned short)(y * height / OUT_HEIGHT);
        s->map_source_width = width; s->map_source_height = height; s->map_width = picture_width;
    }
    uint32_t *buffer = s->buffers[s->current];
    if (s->left[s->current] != left || s->width[s->current] != picture_width) {
        memset(buffer, 0, OUT_BUFFER_SIZE);
        s->left[s->current] = left; s->width[s->current] = picture_width;
    }
    static uint32_t row[OUT_WIDTH];
    int converted = -1;
    for (unsigned y = 0; y < OUT_HEIGHT; y++) {
        if (s->rows[y] != converted) {
            converted = s->rows[y];
            const uint32_t *source = pixels + (size_t)converted * width;
            for (unsigned x = 0; x < picture_width; x++) {
                uint32_t v = source[s->columns[x]];
                row[x] = 0xFF000000u | ((v & 255u) << 16) | (v & 0xFF00u) | ((v >> 16) & 255u);
            }
        }
        memcpy(buffer + (size_t)y * OUT_WIDTH + left, row, picture_width * 4);
    }
    s->drawn = 1;
}

int dhdr_present_exact(DhdrSurface *s, DhdrExact *x, const uint32_t *pixels, uint32_t w, uint32_t h,
                       uint32_t window_w, uint32_t window_h)
{
    (void)window_w; (void)window_h;
    if (!s) return -1;
    if (x) dhdr_exact_submit(x);  /* the scanout, before the picture is replaced or shown */
    if (pixels) {
        if (!w || !h || (uint64_t)w * h > 1024 * 1024) return 0;
        if (x) {  /* into the display texture, as a scanout would: a later capture returns it */
            memcpy(x->display, pixels, (size_t)w * h * 4);
            x->display_width = w; x->display_height = h;
        }
        draw_picture(s, pixels, w, h);
        return 1;
    }
    if (!x || !x->display_width) return 0;
    draw_picture(s, x->display, x->display_width, x->display_height);
    return 1;
}

void dhdr_surface_present(DhdrSurface *s)
{
    if (!s || !s->drawn) return;
    s->drawn = 0;
    s->flips++;
    sceVideoOutSubmitFlip(s->handle, s->current, ORBIS_VIDEO_OUT_FLIP_VSYNC, s->flips);
    s->current = (s->current + 1) % OUT_BUFFERS;
    gnm_frame_done();
}

/* ---- decoded textures and replacements ----
   A decoded texture's texels live in GPU memory (garlic: the CPU writes them once, the pixel stage
   reads them). Replacements (texture packs, the sharp font) are kept on the CPU only: the native
   renderer does not draw them yet (render_ps4.c draws the decoded texture instead). */

struct DhdrTexture {
    volatile int references;
    uint32_t w, h, levels;
    uint32_t *gpu;                   /* the texels (decoded textures), or NULL */
    uint32_t pixels[];               /* replacements' levels (CPU) */
};

DhdrTexture *dhdr_texture_create(const uint32_t *rgba, uint32_t w, uint32_t h)
{
    DhdrTexture *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->references = 1; t->w = w; t->h = h; t->levels = 1;
    t->gpu = gnm_memory_alloc((size_t)w * h * 4, 1);
    if (!t->gpu) { fprintf(stderr, "dhd_render: no GPU memory for a %ux%u texture\n", w, h); free(t); return NULL; }
    memcpy(t->gpu, rgba, (size_t)w * h * 4);
    return t;
}

DhdrTexture *dhdr_replace_create(const uint8_t *rgba_mips, uint32_t w, uint32_t h, uint32_t levels)
{
    size_t bytes = 0;
    for (uint32_t l = 0, lw = w, lh = h; l < levels; l++, lw = lw > 1 ? lw / 2 : 1, lh = lh > 1 ? lh / 2 : 1)
        bytes += (size_t)lw * lh * 4;
    DhdrTexture *t = malloc(sizeof *t + bytes);
    if (!t) return NULL;
    t->references = 1; t->w = w; t->h = h; t->levels = levels; t->gpu = NULL;
    memcpy(t->pixels, rgba_mips, bytes);
    return t;
}

void dhdr_texture_retain(DhdrTexture *t) { if (t) __atomic_add_fetch(&t->references, 1, __ATOMIC_RELAXED); }
void dhdr_texture_release(DhdrTexture *t)
{
    if (t && __atomic_sub_fetch(&t->references, 1, __ATOMIC_ACQ_REL) == 0) {
        gnm_memory_free(t->gpu);  /* the GPU has finished with it: every picture is waited for */
        free(t);
    }
}

/* ---- native: native.wgsl's renderer on the GPU's graphics pipeline ----

   Surfaces: colour ones RGBA8 texels (linear, rows of `pitch` words), depth ones 32-bit float,
   1D-tiled (8x8 micro tiles), as the depth unit requires. A picture is one command buffer: its
   vertices, indices, the draws' constants and imported pixels go into an arena first; then each
   command in order: COPY, IMPORT and ALPHA_UPLOAD as compute dispatches (native_ops.cl), DRAW as
   one draw (or two: the RGB with As / 128, then the alpha alone, when the blend uses As: no
   dual-source blending here) with its target, depth, blend, write mask, scissor and constants;
   then the scanout into the display picture. A full barrier (gnm_full_barrier) parts draws from
   the compute work around them and from draws that read a surface drawn since the last barrier.
   Not yet: conversions (a draw that samples one is skipped), replacements, the Z probe's reads,
   translated VU1 meshes, surface reads. */

enum { NATIVE_SURFACES = 48, ARENA_BYTES = 64u << 20 };

typedef struct {
    uint32_t *mem;
    uint32_t w, h;                   /* texels */
    uint32_t pitch;                  /* colour: words a row; depth: 8x8 tiles a row */
    uint32_t rows;                   /* depth: tile rows */
    uint32_t scale, depth;
} NativeSurface;

/* the draw's constants as native.cl's Nat */
typedef struct {
    DhdrNativeConstants c;
    uint64_t texture;
    uint32_t tex_w, tex_h, tex_pitch, tex_kind, variant, pad;
} NatConstants;
_Static_assert(sizeof(DhdrNativeConstants) == 144, "native.cl's Nat starts with DhdrNativeConstants");

struct DhdrNative {
    NativeSurface surfaces[NATIVE_SURFACES];
    uint8_t *arena;                  /* garlic */
    size_t arena_used;
    uint32_t *scratch;               /* a colour surface's copy (self copies, a draw sampling its target) */
    size_t scratch_bytes;
    uint32_t *display;               /* the displayed picture (onion) */
    uint32_t display_w, display_h;
    GnmShader vs, ps;
    GnmKernel copy, blit, import, alpha, scanout, depth_grow;
    /* what the open command buffer has set */
    int target, depth;
    uint64_t dirty;                  /* surfaces drawn since the last barrier */
    int after_compute;               /* compute work since the last draw: a barrier first */
};

static NativeSurface *native_surface_of(DhdrNative *n, int index)
{
    return index >= 0 && index < NATIVE_SURFACES && n->surfaces[index].mem ? &n->surfaces[index] : NULL;
}

DhdrNative *dhdr_native_create(uint32_t flags)
{
    (void)flags;  /* no dual-source blending here */
    if (device_made <= 0) return NULL;
    DhdrNative *n = calloc(1, sizeof *n);
    if (!n) return NULL;
    int ok = gnm_graphics_shader("native_vs", &n->vs) && gnm_graphics_shader("native_ps", &n->ps);
    ok = ok && gnm_kernel("native_copy", &n->copy) && gnm_kernel("native_blit", &n->blit) &&
         gnm_kernel("native_import", &n->import) && gnm_kernel("native_alpha", &n->alpha) &&
         gnm_kernel("native_scanout", &n->scanout) && gnm_kernel("native_depth_grow", &n->depth_grow);
    n->arena = ok ? gnm_memory_alloc(ARENA_BYTES, 1) : NULL;
    n->display = ok ? gnm_memory_alloc(4096u * 4096u * 4u, 0) : NULL;
    if (!ok || !n->arena || !n->display) {
        fprintf(stderr, "dhd_render: the native renderer's shaders or memory are missing\n");
        gnm_memory_free(n->arena);
        gnm_memory_free(n->display);
        free(n);
        return NULL;
    }
    fprintf(stderr, "dhd_render: native renderer on the GPU's graphics pipeline (pixel stage %u VGPRs)\n",
            ((n->ps.rsrc1 & 63u) + 1u) * 4u);
    return n;
}

void dhdr_native_destroy(DhdrNative *n)
{
    if (!n) return;
    for (int i = 0; i < NATIVE_SURFACES; i++) gnm_memory_free(n->surfaces[i].mem);
    gnm_memory_free(n->arena);
    gnm_memory_free(n->scratch);
    gnm_memory_free(n->display);
    free(n);
}

static void dispatch2d(const GnmKernel *k, const void *args, uint32_t size, uint32_t w, uint32_t h)
{
    if (w && h) gnm_dispatch(k, args, size, (w + 7) / 8, (h + 7) / 8, 8, 8);
}

int dhdr_native_surface(DhdrNative *n, uint32_t index, uint32_t scale, uint32_t w, uint32_t h, uint32_t depth,
                        uint32_t size_out[2])
{
    if (index >= NATIVE_SURFACES || !w || !h || w > 16384 || h > 16384) return 0;
    NativeSurface *old = &n->surfaces[index];
    if (old->mem && old->scale == scale && old->depth == depth && old->w >= w && old->h >= h) {
        size_out[0] = old->w; size_out[1] = old->h;
        return 1;
    }
    NativeSurface s = *old;
    int keep = old->mem && old->scale == scale && old->depth == depth;
    s.w = keep && old->w > w ? old->w : w;
    s.h = keep && old->h > h ? old->h : h;
    s.scale = scale; s.depth = depth;
    size_t bytes;
    if (depth) {
        s.pitch = (s.w + 7) / 8; s.rows = (s.h + 7) / 8;
        bytes = (size_t)s.pitch * s.rows * 256;
    } else {
        s.pitch = (s.w + 63) & ~63u; s.rows = (s.h + 7) & ~7u;
        bytes = (size_t)s.pitch * s.rows * 4;
    }
    s.mem = gnm_memory_alloc(bytes, 1);
    if (!s.mem) { fprintf(stderr, "dhd_render: no GPU memory for surface %u (%ux%u)\n", index, s.w, s.h); return 0; }
    memset(s.mem, 0, bytes);  /* new texels zero (depth 0: the farthest) */
    if (keep) {  /* the old contents, by the GPU (garlic is slow to read from the CPU) */
        gnm_begin();
        if (depth) {
            struct { uint64_t from; uint32_t tx, ty; uint64_t to; uint32_t to_tx, pad; } a =
                {(uint64_t)(uintptr_t)old->mem, old->pitch, old->rows, (uint64_t)(uintptr_t)s.mem, s.pitch, 0};
            gnm_dispatch(&n->depth_grow, &a, sizeof a, old->pitch * old->rows, 1, 64, 1);
        } else {
            struct { uint64_t from; uint32_t from_pitch, pad0; uint64_t to; uint32_t to_pitch, sx, sy, dx, dy, w, h; } a =
                {(uint64_t)(uintptr_t)old->mem, old->pitch, 0, (uint64_t)(uintptr_t)s.mem, s.pitch, 0, 0, 0, 0, old->w, old->h};
            dispatch2d(&n->copy, &a, sizeof a, old->w, old->h);
        }
        gnm_finish();
    }
    gnm_memory_free(old->mem);
    *old = s;
    size_out[0] = s.w; size_out[1] = s.h;
    return 1;
}

/* memory in the picture's arena (256-byte aligned); NULL when it is full */
static void *arena_take(DhdrNative *n, size_t bytes)
{
    size_t at = (n->arena_used + 255) & ~(size_t)255;
    if (at + bytes > ARENA_BYTES) return NULL;
    n->arena_used = at + bytes;
    return n->arena + at;
}

static void barrier_if(DhdrNative *n, int needed)
{
    if (!needed) return;
    gnm_full_barrier();
    n->dirty = 0;
    n->after_compute = 0;
}

/* a texel rectangle of one colour surface to another through the compute queue */
static void native_copy_rect(DhdrNative *n, const uint32_t *from, uint32_t from_pitch, uint32_t *to, uint32_t to_pitch,
                             uint32_t sx, uint32_t sy, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h)
{
    struct { uint64_t from; uint32_t from_pitch, pad0; uint64_t to; uint32_t to_pitch, sx, sy, dx, dy, w, h; } a =
        {(uint64_t)(uintptr_t)from, from_pitch, 0, (uint64_t)(uintptr_t)to, to_pitch, sx, sy, dx, dy, w, h};
    dispatch2d(&n->copy, &a, sizeof a, w, h);
}

static int scratch_for(DhdrNative *n, const NativeSurface *s)
{
    size_t bytes = (size_t)s->pitch * s->rows * 4;
    if (n->scratch_bytes >= bytes) return 1;
    gnm_memory_free(n->scratch);
    n->scratch = gnm_memory_alloc(bytes, 1);
    n->scratch_bytes = n->scratch ? bytes : 0;
    return n->scratch != NULL;
}

static void transfer(DhdrNative *n, const DhdrCommand *k, DhdrPictureResult *out)
{
    NativeSurface *t = native_surface_of(n, k->target);
    if (!t || t->depth) return;
    barrier_if(n, 1);  /* the draws before, finished and in memory */
    if (k->kind == DHDR_CMD_ALPHA_UPLOAD) {
        if (!k->pixels) return;
        uint32_t *page = arena_take(n, 64 * 32 * 4);
        if (!page) return;
        memcpy(page, k->pixels, 64 * 32 * 4);
        struct { uint64_t page; uint32_t x0, y0, s, pad0; uint64_t to; uint32_t to_pitch, to_w, to_h, pad1; } a =
            {(uint64_t)(uintptr_t)page, (uint32_t)k->x, (uint32_t)k->y, t->scale, 0, (uint64_t)(uintptr_t)t->mem, t->pitch, t->w, t->h, 0};
        dispatch2d(&n->alpha, &a, sizeof a, 64 * t->scale, 32 * t->scale);
    } else if (k->kind == DHDR_CMD_IMPORT) {
        if (!k->pixels || k->w <= 0 || k->h <= 0 || k->x < 0 || k->y < 0) return;
        size_t bytes = (size_t)k->w * k->h * 4;
        uint32_t *pixels = arena_take(n, bytes);
        if (!pixels) return;
        memcpy(pixels, k->pixels, bytes);
        struct { uint64_t pixels; uint32_t w, h, x0, y0, s, pad0; uint64_t to; uint32_t to_pitch, to_w, to_h, pad1; } a =
            {(uint64_t)(uintptr_t)pixels, (uint32_t)k->w, (uint32_t)k->h, (uint32_t)k->x, (uint32_t)k->y, t->scale, 0,
             (uint64_t)(uintptr_t)t->mem, t->pitch, t->w, t->h, 0};
        dispatch2d(&n->import, &a, sizeof a, (uint32_t)k->w * t->scale, (uint32_t)k->h * t->scale);
        out->imports++;
        out->import_px += (uint64_t)k->w * k->h;
    } else {  /* COPY */
        NativeSurface *s = native_surface_of(n, k->source);
        if (!s || s->depth || !s->scale || !t->scale) return;
        if (s->scale != t->scale) {
            uint32_t ts = t->scale;
            int32_t dx = k->x * (int32_t)ts, dy = k->y * (int32_t)ts, w = k->w * (int32_t)ts, h = k->h * (int32_t)ts;
            if (dx < 0 || dy < 0 || w <= 0 || h <= 0) return;
            if ((uint32_t)(dx + w) > t->w) w = (int32_t)t->w - dx;
            if ((uint32_t)(dy + h) > t->h) h = (int32_t)t->h - dy;
            if (w <= 0 || h <= 0) return;
            struct { uint64_t from; uint32_t from_pitch, from_w, from_h, pad0; uint64_t to; uint32_t to_pitch, dx, dy, w, h;
                     float ox, oy, t, s; } a =
                {(uint64_t)(uintptr_t)s->mem, s->pitch, s->w, s->h, 0, (uint64_t)(uintptr_t)t->mem, t->pitch,
                 (uint32_t)dx, (uint32_t)dy, (uint32_t)w, (uint32_t)h, (float)(k->sx - k->x), (float)(k->sy - k->y),
                 (float)t->scale, (float)s->scale};
            dispatch2d(&n->blit, &a, sizeof a, (uint32_t)w, (uint32_t)h);
            n->after_compute = 1;
            return;
        }
        int32_t ss = (int32_t)s->scale, w = k->w, h = k->h;
        int32_t sw = (int32_t)s->w / ss, sh = (int32_t)s->h / ss, tw = (int32_t)t->w / ss, th = (int32_t)t->h / ss;
        if (k->sx + w > sw) w = sw - k->sx;
        if (k->sy + h > sh) h = sh - k->sy;
        if (k->x + w > tw) w = tw - k->x;
        if (k->y + h > th) h = th - k->y;
        if (w <= 0 || h <= 0 || k->sx < 0 || k->sy < 0 || k->x < 0 || k->y < 0) return;
        out->copies++;
        out->copy_px += (uint64_t)w * h;
        uint32_t bx = (uint32_t)(k->sx * ss), by = (uint32_t)(k->sy * ss), bw = (uint32_t)(w * ss), bh = (uint32_t)(h * ss);
        uint32_t dx = (uint32_t)(k->x * ss), dy = (uint32_t)(k->y * ss);
        if (s == t) {  /* through the scratch copy: the rectangles may overlap */
            if (!scratch_for(n, s)) return;
            native_copy_rect(n, s->mem, s->pitch, n->scratch, s->pitch, bx, by, bx, by, bw, bh);
            gnm_barrier();
            native_copy_rect(n, n->scratch, s->pitch, t->mem, t->pitch, bx, by, dx, dy, bw, bh);
            out->self_copies++;
            out->self_copy_px += (uint64_t)w * h;
        } else {
            native_copy_rect(n, s->mem, s->pitch, t->mem, t->pitch, bx, by, dx, dy, bw, bh);
        }
    }
    n->after_compute = 1;
}

/* D3D11 blend factor -> GCN BlendOp; operation -> CombFunc */
static uint32_t blend_factor(uint32_t d3d)
{
    switch (d3d) {
    case 1: return 0;   /* ZERO */
    case 2: return 1;   /* ONE */
    case 5: return 4;   /* SRC_ALPHA */
    case 6: return 5;   /* ONE_MINUS_SRC_ALPHA */
    case 7: return 6;   /* DST_ALPHA */
    case 8: return 7;   /* ONE_MINUS_DST_ALPHA */
    case 14: return 13; /* CONSTANT_COLOR */
    case 15: return 14; /* ONE_MINUS_CONSTANT_COLOR */
    default: return 1;
    }
}
static uint32_t blend_comb(uint32_t op)
{
    switch (op) {
    case 2: return 1;   /* SUBTRACT: src - dst */
    case 3: return 4;   /* REV_SUBTRACT: dst - src */
    case 4: return 2;   /* MIN */
    case 5: return 3;   /* MAX */
    default: return 0;  /* ADD */
    }
}
/* the key's depth field (16 has depth, 8 write, 0-7 the function) as DB_DEPTH_CONTROL */
static uint32_t depth_control(uint32_t depth)
{
    if (!(depth & 16)) return 0;
    uint32_t f = depth & 7, func = f == 0 ? 0u : f == 2 ? 6u : f == 3 ? 4u : f == 4 ? 2u : 7u;  /* NEVER, GEQUAL, GREATER, EQUAL, ALWAYS */
    return (1u << 1) | ((depth & 8) ? 1u << 2 : 0) | func << 4;
}

/* the target (and depth) of the draws that follow, the viewport over the target */
static void bind_targets(DhdrNative *n, int target, int depth)
{
    if (n->target == target && n->depth == depth) return;
    NativeSurface *t = &n->surfaces[target];
    uint64_t rt = (uint64_t)(uintptr_t)t->mem;
    uint32_t cb[7] = {(uint32_t)(rt >> 8), t->pitch / 8 - 1, t->pitch * t->rows / 64 - 1, 0, 10u << 2, 8, 0};
    gnm_context_regs(0x318, cb, 6);              /* CB_COLOR0_BASE .. ATTRIB: RGBA8 UNORM, LinearAligned */
    uint32_t screen[2] = {0, t->w | t->h << 16};
    gnm_context_regs(0x00C, screen, 2);          /* PA_SC_SCREEN_SCISSOR */
    uint32_t window[2] = {0x80000000u, t->w | t->h << 16};
    gnm_context_regs(0x081, window, 2);          /* PA_SC_WINDOW_SCISSOR */
    uint32_t vport_scissor[2] = {0x80000000u, t->w | t->h << 16};
    gnm_context_regs(0x094, vport_scissor, 2);   /* PA_SC_VPORT_SCISSOR_0 */
    float vp[6] = {t->w * 0.5f, t->w * 0.5f, t->h * -0.5f, t->h * 0.5f, 1.0f, 0.0f};
    gnm_context_regs(0x10F, (const uint32_t *)vp, 6);  /* PA_CL_VPORT_XSCALE .. ZOFFSET */
    if (depth >= 0) {
        NativeSurface *d = &n->surfaces[depth];
        uint64_t db = (uint64_t)(uintptr_t)d->mem;
        gnm_context_reg(0x00F, (2u << 4) | (12u << 8));     /* DB_DEPTH_INFO: 1D thin */
        uint32_t z[8] = {3u, 1u << 29, (uint32_t)(db >> 8), (uint32_t)(db >> 8), (uint32_t)(db >> 8), (uint32_t)(db >> 8),
                         (d->pitch - 1) | ((d->rows - 1) << 11), d->pitch * d->rows - 1};
        gnm_context_regs(0x010, z, 8);                      /* DB_Z_INFO .. DB_DEPTH_SLICE: Z_32_FLOAT */
    } else {
        gnm_context_reg(0x010, 0);                          /* DB_Z_INFO: no depth buffer */
        gnm_context_reg(0x011, 0);
    }
    n->target = target;
    n->depth = depth;
}

/* one draw of k's vertices with this state (one pass of draw_passes) */
static void native_draw(DhdrNative *n, const DhdrCommand *k, const NatConstants *nat_in, uint32_t blend_key, uint32_t mask,
                        uint32_t depth, uint32_t variant, uint64_t vertices, uint64_t indices, int indexed,
                        const uint32_t scissor[4])
{
    NatConstants *nat = arena_take(n, sizeof *nat);
    if (!nat) return;
    *nat = *nat_in;
    nat->variant = variant;
    uint32_t blend = 0;
    if ((blend_key & 1) && variant != 2) {
        uint32_t src = blend_factor(blend_key >> 1 & 31), dst = blend_factor(blend_key >> 6 & 31);
        uint32_t comb = blend_comb(blend_key >> 11 & 7);
        blend = src | comb << 5 | dst << 8 | 1u << 16 | 0u << 21 | 0u << 24 | 1u << 29 | 1u << 30;  /* alpha: ONE, ZERO */
    }
    gnm_context_reg(0x1E0, blend);                         /* CB_BLEND0_CONTROL */
    gnm_context_reg(0x08E, mask & 15);                     /* CB_TARGET_MASK */
    gnm_context_reg(0x200, depth_control(depth));          /* DB_DEPTH_CONTROL */
    if (blend) {
        float fix = (float)k->blend_fix / 128.0f;
        float constant[4] = {fix, fix, fix, fix};
        gnm_context_regs(0x105, (const uint32_t *)constant, 4);  /* CB_BLEND_RED .. ALPHA */
    }
    uint32_t tl = 0x80000000u | scissor[0] | scissor[1] << 16, br = (scissor[0] + scissor[2]) | (scissor[1] + scissor[3]) << 16;
    uint32_t generic[2] = {tl, br};
    gnm_context_regs(0x090, generic, 2);                   /* PA_SC_GENERIC_SCISSOR */
    uint64_t constants = (uint64_t)(uintptr_t)nat;
    uint32_t vs_user[10] = {0, 0, (uint32_t)vertices, (uint32_t)(vertices >> 32), (uint32_t)indices, (uint32_t)(indices >> 32),
                            (uint32_t)constants, (uint32_t)(constants >> 32), k->first, (uint32_t)indexed};
    gnm_graphics_sh(0x4C, vs_user, 10);                    /* SPI_SHADER_USER_DATA_VS_0 .. 9 */
    uint32_t ps_user[4] = {0, 0, (uint32_t)constants, (uint32_t)(constants >> 32)};
    gnm_graphics_sh(0x0C, ps_user, 4);                     /* SPI_SHADER_USER_DATA_PS_0 .. 3 */
    gnm_draw_auto(k->count);
}

/* the stages and the state every draw shares */
static void draw_state(DhdrNative *n)
{
    gnm_draw_state_init();
    uint32_t vs[4] = {(uint32_t)(n->vs.address >> 8), (uint32_t)(n->vs.address >> 40), n->vs.rsrc1, n->vs.rsrc2 | (10u << 1)};
    gnm_graphics_sh(0x48, vs, 4);                          /* SPI_SHADER_PGM_LO/HI/RSRC1/RSRC2_VS: 10 user SGPRs */
    uint32_t ps[4] = {(uint32_t)(n->ps.address >> 8), (uint32_t)(n->ps.address >> 40), n->ps.rsrc1, n->ps.rsrc2 | (4u << 1)};
    gnm_graphics_sh(0x08, ps, 4);                          /* SPI_SHADER_PGM_LO/HI/RSRC1/RSRC2_PS: 4 user SGPRs */
    gnm_context_reg(0x1B1, 1u << 1);                       /* SPI_VS_OUT_CONFIG: two parameters */
    gnm_context_reg(0x1C3, 4);                             /* SPI_SHADER_POS_FORMAT: POS0 4COMP */
    uint32_t inputs[2] = {0, 1};
    gnm_context_regs(0x191, inputs, 2);                    /* SPI_PS_INPUT_CNTL_0/1: parameters 0, 1 */
    gnm_context_reg(0x1B3, n->ps.ps_input_ena);            /* SPI_PS_INPUT_ENA */
    gnm_context_reg(0x1B4, n->ps.ps_input_addr);           /* SPI_PS_INPUT_ADDR */
    gnm_context_reg(0x1B6, 2);                             /* SPI_PS_IN_CONTROL: NUM_INTERP 2 */
    gnm_context_reg(0x1B8, 0);                             /* SPI_BARYC_CNTL */
    gnm_context_reg(0x1C4, 0);                             /* SPI_SHADER_Z_FORMAT */
    gnm_context_reg(0x1C5, 4);                             /* SPI_SHADER_COL_FORMAT: FP16_ABGR */
    gnm_context_reg(0x203, (1u << 4) | (1u << 6));         /* DB_SHADER_CONTROL: EARLY_Z_THEN_LATE_Z, KILL_ENABLE */
    gnm_context_reg(0x202, 0x00CC0010u);                   /* CB_COLOR_CONTROL: normal, copy */
    gnm_context_reg(0x08F, 0xF);                           /* CB_SHADER_MASK */
    gnm_context_reg(0x206, 0x43Fu);                        /* PA_CL_VTE_CNTL: viewport scale and offset, W0 */
    gnm_context_reg(0x204, 1u << 19);                      /* PA_CL_CLIP_CNTL: DX clip space, clipping on */
    gnm_context_reg(0x205, 0);                             /* PA_SU_SC_MODE_CNTL: no culling */
    gnm_context_reg(0x207, 0);                             /* PA_CL_VS_OUT_CNTL */
    gnm_context_reg(0x292, 0);                             /* PA_SC_MODE_CNTL_0 */
    gnm_context_reg(0x2F8, 0);                             /* PA_SC_AA_CONFIG */
    gnm_context_reg(0x080, 0);                             /* PA_SC_WINDOW_OFFSET */
    float zrange[2] = {0.0f, 1.0f};
    gnm_context_regs(0x0B4, (const uint32_t *)zrange, 2);  /* PA_SC_VPORT_ZMIN/ZMAX_0 */
    gnm_context_reg(0x002, 0);                             /* DB_DEPTH_VIEW */
    gnm_context_reg(0x2AF, 0);                             /* DB_HTILE_SURFACE */
    gnm_context_reg(0x000, 0);                             /* DB_RENDER_CONTROL */
    gnm_context_reg(0x003, 0);                             /* DB_RENDER_OVERRIDE */
    gnm_context_reg(0x2D5, 0);                             /* VGT_SHADER_STAGES_EN */
    gnm_context_reg(0x290, 0);                             /* VGT_GS_MODE */
    gnm_uconfig_reg(0x242, 4);                             /* VGT_PRIMITIVE_TYPE: triangle list */
    n->target = n->depth = -2;
}

static int native_render(DhdrNative *n, const DhdrPicture *p, const uint32_t *indices, uint32_t index_count,
                         DhdrPictureResult *out)
{
    memset(out, 0, sizeof *out);
    n->arena_used = 0;
    size_t vertex_bytes = (size_t)p->vertex_count * sizeof(DhdrVertex);
    void *vertices = arena_take(n, vertex_bytes ? vertex_bytes : 4);
    void *index_copy = indices ? arena_take(n, (size_t)index_count * 4 + 4) : NULL;
    if (!vertices || (indices && !index_copy)) { fprintf(stderr, "dhd_render: a picture too large for its arena\n"); return 0; }
    if (vertex_bytes) memcpy(vertices, p->vertices, vertex_bytes);
    if (indices) memcpy(index_copy, indices, (size_t)index_count * 4);
    uint64_t vertex_at = (uint64_t)(uintptr_t)vertices, index_at = (uint64_t)(uintptr_t)index_copy;
    gnm_begin();
    draw_state(n);
    n->dirty = 0;
    n->after_compute = 0;
    for (uint32_t i = 0; i < p->command_count; i++) {
        const DhdrCommand *k = &p->commands[i];
        if (!gnm_room()) {  /* a full buffer goes now; the rest follows in another */
            gnm_finish();
            gnm_begin();
            draw_state(n);
        }
        if (k->kind != DHDR_CMD_DRAW) { transfer(n, k, out); continue; }
        if (!k->count) continue;
        NativeSurface *target = native_surface_of(n, k->target);
        if (!target || target->depth) continue;
        int depth = -1;
        NativeSurface *d = (k->depth >= 0 && (k->depth_key & 8)) ? native_surface_of(n, k->depth) : NULL;
        if (d && d->depth && d->w == target->w && d->h == target->h) depth = k->depth;
        NatConstants nat;
        memset(&nat, 0, sizeof nat);
        nat.c = k->constants;
        nat.c.flags &= ~2048u;        /* replacements are not drawn yet: the decoded texture is */
        uint32_t source_scale = 1;
        if (k->texture_surface >= 0) {
            NativeSurface *s = native_surface_of(n, k->texture_surface);
            if (!s || s->depth) continue;
            source_scale = s->scale;
            const uint32_t *texels = s->mem;
            if (k->texture_surface == k->target) {  /* sampling the target: from a copy of it */
                if (!scratch_for(n, s)) continue;
                barrier_if(n, 1);
                native_copy_rect(n, s->mem, s->pitch, n->scratch, s->pitch, 0, 0, 0, 0, s->w, s->h);
                barrier_if(n, 1);
                texels = n->scratch;
            } else {
                barrier_if(n, (n->dirty >> k->texture_surface & 1) || n->after_compute);
            }
            nat.texture = (uint64_t)(uintptr_t)texels;
            nat.tex_w = s->w; nat.tex_h = s->h; nat.tex_pitch = s->pitch; nat.tex_kind = 1;
        } else if (k->texture && k->texture->gpu) {
            nat.texture = (uint64_t)(uintptr_t)k->texture->gpu;
            nat.tex_w = k->texture->w; nat.tex_h = k->texture->h; nat.tex_pitch = k->texture->w; nat.tex_kind = 2;
        } else if (k->convert >= 0) {
            continue;  /* conversions: not yet */
        }
        barrier_if(n, n->after_compute);
        uint32_t t = target->scale ? target->scale : 1;
        nat.c.scale = t | source_scale << 8 | (source_scale < t ? source_scale : t) << 16;
        nat.c.target[0] = (float)target->w / (float)t;
        nat.c.target[1] = (float)target->h / (float)t;
        int32_t left = k->scissor[0] * (int32_t)t, top = k->scissor[1] * (int32_t)t;
        int32_t right = k->scissor[2] * (int32_t)t, bottom = k->scissor[3] * (int32_t)t;
        if (left < 0) left = 0;
        if (top < 0) top = 0;
        if (right > (int32_t)target->w) right = (int32_t)target->w;
        if (bottom > (int32_t)target->h) bottom = (int32_t)target->h;
        if (right <= left || bottom <= top) continue;
        uint32_t scissor[4] = {(uint32_t)left, (uint32_t)top, (uint32_t)(right - left), (uint32_t)(bottom - top)};
        /* draw_passes */
        uint32_t blend_key = k->blend_key, mask = k->color_mask;
        uint32_t src = blend_key >> 1 & 31, dst = blend_key >> 6 & 31, op = blend_key >> 11 & 7;
        if ((blend_key & 1) && src == 1 && dst == 2 && (op == 0 || op == 1 || op == 3)) { blend_key = 0; mask &= 8; }
        src = blend_key >> 1 & 31; dst = blend_key >> 6 & 31;
        uint32_t dk = k->depth_key;
        int uses_as = (blend_key & 1) && (src == 5 || src == 6 || dst == 5 || dst == 6);
        uint32_t zfield = depth >= 0 ? 16u | ((dk & 4) ? 8u : 0u) | (dk & 3) : 0;
        if (!mask && !(zfield & 8)) continue;
        bind_targets(n, k->target, depth);
        if (!uses_as) {
            native_draw(n, k, &nat, blend_key, mask, zfield, 0, vertex_at, index_at, indices != NULL, scissor);
        } else {
            if (mask & 7) native_draw(n, k, &nat, blend_key, mask & 7, zfield, 1, vertex_at, index_at, indices != NULL, scissor);
            if (mask & 8) {
                uint32_t second = depth >= 0 ? 16u | ((dk & 4) ? 4u : (dk & 3)) : 0;  /* no write; EQUAL after a write */
                native_draw(n, k, &nat, 0, 8, second, 2, vertex_at, index_at, indices != NULL, scissor);
                out->two_pass_draws++;
            }
        }
        out->draws++;
        n->dirty |= 1ull << k->target;
        if (depth >= 0) n->dirty |= 1ull << depth;
    }
    /* the scanout (or the movie's picture, or black) */
    int shown = 0;
    if (!p->movie_on && p->display_surface >= 0) {
        NativeSurface *s = native_surface_of(n, p->display_surface);
        uint32_t sc = s && s->scale ? s->scale : 1;
        uint32_t w = p->display[5] * sc, h = p->display[6] * sc;
        if (s && !s->depth && w && h && w <= 4096 && h <= 4096) {
            barrier_if(n, 1);
            struct { uint64_t from; uint32_t from_pitch, from_w, from_h, pad0; uint64_t to; uint32_t width, height, ox, oy, interlace, s; } a =
                {(uint64_t)(uintptr_t)s->mem, s->pitch, s->w, s->h, 0, (uint64_t)(uintptr_t)n->display, w, h,
                 p->display[3], p->display[4], p->display[7], sc};
            dispatch2d(&n->scanout, &a, sizeof a, w, h);
            n->display_w = w; n->display_h = h;
            shown = 1;
        }
    }
    int ok = gnm_finish();
    if (p->movie_on && p->movie && p->movie_width && p->movie_height &&
        (uint64_t)p->movie_width * p->movie_height <= 4096u * 4096u) {
        memcpy(n->display, p->movie, (size_t)p->movie_width * p->movie_height * 4);
        n->display_w = p->movie_width; n->display_h = p->movie_height;
        shown = 1;
    } else if (p->display_surface == -2 && n->display_w) {
        memset(n->display, 0, (size_t)n->display_w * n->display_h * 4);  /* the display is off: black */
        shown = 1;
    }
    out->shown = (uint32_t)shown;
    out->display_width = n->display_w;
    out->display_height = n->display_h;
    return ok;
}

int dhdr_native_picture(DhdrNative *n, const DhdrPicture *p, DhdrPictureResult *out)
{
    return native_render(n, p, NULL, 0, out);
}

int dhdr_native_picture_indexed(DhdrNative *n, const DhdrPicture *p, const uint32_t *indices, uint32_t index_count,
                                DhdrPictureResult *out)
{
    return native_render(n, p, indices, index_count, out);
}

int dhdr_native_present(DhdrNative *n, DhdrSurface *s, const DhdrPresentRect *r)
{
    (void)r;  /* the picture fills the screen's height in the widescreen setting's shape (draw_picture) */
    if (!s) return -1;
    if (!n->display_w) return 0;
    draw_picture(s, n->display, n->display_w, n->display_h);
    return 1;
}

int dhdr_native_read_begin(DhdrNative *n, uint32_t *w, uint32_t *h)
{
    *w = n->display_w; *h = n->display_h;
    return n->display_w != 0;
}
int dhdr_native_read_finish(DhdrNative *n, uint32_t *pixels)
{
    memcpy(pixels, n->display, (size_t)n->display_w * n->display_h * 4);
    return 1;
}
void dhdr_native_options(DhdrNative *n, uint32_t flags) { (void)n; (void)flags; }
int dhdr_native_zread_start(DhdrNative *n, uint32_t index, const uint32_t rect[4], uint64_t tag)
{
    (void)n; (void)index; (void)rect; (void)tag;
    return 0;
}
int dhdr_native_zread_take(DhdrNative *n, uint32_t *out, uint32_t cap, uint32_t rect_out[4], uint64_t *tag_out)
{
    (void)n; (void)out; (void)cap; (void)rect_out; (void)tag_out;
    return 0;
}
int dhdr_native_meshes(void) { return 0; }
void dhdr_native_pipeline_list(const char *path) { (void)path; }
const char *dhdr_native_pipeline_crash(uint64_t *key) { *key = 0; return NULL; }
int dhdr_native_surface_read_begin(DhdrNative *n, uint32_t index, const uint32_t rect[4], uint32_t size[2])
{
    (void)n; (void)index; (void)rect; size[0] = size[1] = 0;
    return 0;
}
int dhdr_native_surface_read_finish(DhdrNative *n, uint32_t *pixels) { (void)n; (void)pixels; return 0; }

/* ---- texture packs' PNG files: not decoded on the PS4 yet (no packs) ---- */

int dhdr_png_size(const uint8_t *data, size_t size, uint32_t *w, uint32_t *h)
{
    (void)data; (void)size; *w = *h = 0;
    return 0;
}
int dhdr_png_decode(const uint8_t *data, size_t size, uint8_t *rgba, uint32_t w, uint32_t h, uint32_t opaque)
{
    (void)data; (void)size; (void)rgba; (void)w; (void)h; (void)opaque;
    return 0;
}
