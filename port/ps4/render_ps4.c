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
   - Native (dhdr_native_*): not yet (the graphics pipeline's checks pass: gnm.c); create answers
     NULL, so gpu_rust.c keeps the exact renderer. Decoded textures are kept for when it comes. */
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
uint32_t dhdr_device_max_texture(void) { return 16384; }
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

/* ---- decoded textures and replacements (kept for the native renderer) ---- */

struct DhdrTexture {
    volatile int references;
    uint32_t w, h, levels;
    uint32_t pixels[];
};

static DhdrTexture *texture_make(const void *rgba, uint32_t w, uint32_t h, uint32_t levels, size_t bytes)
{
    DhdrTexture *t = malloc(sizeof *t + bytes);
    if (!t) { fprintf(stderr, "dhd_render: out of memory for a %ux%u texture\n", w, h); return NULL; }
    t->references = 1; t->w = w; t->h = h; t->levels = levels;
    memcpy(t->pixels, rgba, bytes);
    return t;
}

DhdrTexture *dhdr_texture_create(const uint32_t *rgba, uint32_t w, uint32_t h)
{
    return texture_make(rgba, w, h, 1, (size_t)w * h * 4);
}

DhdrTexture *dhdr_replace_create(const uint8_t *rgba_mips, uint32_t w, uint32_t h, uint32_t levels)
{
    size_t bytes = 0;
    for (uint32_t l = 0, lw = w, lh = h; l < levels; l++, lw = lw > 1 ? lw / 2 : 1, lh = lh > 1 ? lh / 2 : 1)
        bytes += (size_t)lw * lh * 4;
    return texture_make(rgba_mips, w, h, levels, bytes);
}

void dhdr_texture_retain(DhdrTexture *t) { if (t) __atomic_add_fetch(&t->references, 1, __ATOMIC_RELAXED); }
void dhdr_texture_release(DhdrTexture *t)
{
    if (t && __atomic_sub_fetch(&t->references, 1, __ATOMIC_ACQ_REL) == 0) free(t);
}

/* ---- native: not yet on the PS4 (gpu_rust.c then keeps the exact renderer) ---- */

DhdrNative *dhdr_native_create(uint32_t flags)
{
    (void)flags;
    fprintf(stderr, "dhd_render: the native renderer is not on the PS4 yet\n");
    return NULL;
}
void dhdr_native_destroy(DhdrNative *n) { (void)n; }
int dhdr_native_surface(DhdrNative *n, uint32_t index, uint32_t scale, uint32_t w, uint32_t h, uint32_t depth,
                        uint32_t size_out[2])
{
    (void)n; (void)index; (void)scale; (void)w; (void)h; (void)depth; size_out[0] = size_out[1] = 0;
    return 0;
}
int dhdr_native_picture(DhdrNative *n, const DhdrPicture *p, DhdrPictureResult *out)
{
    (void)n; (void)p; memset(out, 0, sizeof *out);
    return 0;
}
int dhdr_native_picture_indexed(DhdrNative *n, const DhdrPicture *p, const uint32_t *indices, uint32_t index_count,
                                DhdrPictureResult *out)
{
    (void)indices; (void)index_count;
    return dhdr_native_picture(n, p, out);
}
int dhdr_native_present(DhdrNative *n, DhdrSurface *s, const DhdrPresentRect *r) { (void)n; (void)s; (void)r; return 0; }
int dhdr_native_read_begin(DhdrNative *n, uint32_t *w, uint32_t *h) { (void)n; *w = *h = 0; return 0; }
int dhdr_native_read_finish(DhdrNative *n, uint32_t *pixels) { (void)n; (void)pixels; return 0; }
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
