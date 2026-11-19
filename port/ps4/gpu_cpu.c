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

#include <orbis/SystemService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include "../gs/gs.h"
#include "../gs/gpu.h"
#include "../gs/perf_log.h"
#include "gnm.h"

double host_widescreen_aspect(void);  /* port/hle/widescreen.c: the picture's shape, 0 for 4:3 */

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

/* ---- raster.wgsl in C ---- */

static inline uint32_t load_word(const uint32_t *memory, uint32_t word)
{
    return __atomic_load_n(&memory[word], __ATOMIC_RELAXED);
}

/* D3D's ftoi/ftou: NaN gives 0 and out-of-range values saturate. */
static inline int32_t f2i(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, 4);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0;
    if (x >= 2147483648.0f) return 2147483647;
    if (x <= -2147483648.0f) return (int32_t)(-2147483647 - 1);
    return (int32_t)x;
}
static inline uint32_t f2u(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, 4);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u || x <= 0.0f) return 0;
    if (x >= 4294967296.0f) return 0xFFFFFFFFu;
    return (uint32_t)x;
}

static inline uint32_t bits_per_pixel(uint32_t psm)
{
    if (psm == 1u || psm == 49u) return 24u;
    if ((psm & 7u) == 2u) return 16u;
    if (psm == 19u || psm == 27u) return 8u;
    if (psm == 20u || psm == 36u || psm == 44u) return 4u;
    return 32u;
}
static inline uint32_t pixel_mask(uint32_t bits) { return bits == 32u ? 0xFFFFFFFFu : (1u << bits) - 1u; }

/* The bit address of pixel (x, y) in GS memory (the shader's ADDRESS_SHIFT form). */
static inline uint32_t gs_address(uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
{
    uint32_t pw = 64u, ph = 32u, bits = 32u, high = 0u, block, column, bx, by;
    if (psm == 19u || psm == 20u) { pw = 128u; ph = psm == 19u ? 64u : 128u; }
    else if ((psm & 7u) == 2u) ph = 64u;
    uint32_t sx = pw == 128u ? 7u : 6u;
    uint32_t sy = ph == 128u ? 7u : ph == 64u ? 6u : 5u;
    uint32_t px = x, py = y;
    if (psm == 19u) {
        bits = 8u; bx = px / 16u; by = py / 16u;
        block = (bx & 1u) | ((by & 1u) << 1u) | ((bx & 2u) << 1u) | ((by & 2u) << 2u) | ((bx & 4u) << 2u);
        column = ((px & 1u) << 2u) | ((px & 2u) << 3u) | ((px & 4u) << 3u) | ((px & 8u) >> 2u) |
                 ((py & 1u) << 3u) | ((py & 2u) >> 1u) | ((py & 4u) << 4u) | ((py & 8u) << 4u);
        column ^= (((py >> 1u) ^ (py >> 2u)) & 1u) << 5u;
    } else if (psm == 20u) {
        bits = 4u; bx = px / 32u; by = py / 16u;
        block = (by & 1u) | ((bx & 1u) << 1u) | ((by & 2u) << 1u) | ((bx & 2u) << 2u) | ((by & 4u) << 2u);
        column = ((px & 1u) << 3u) | ((px & 2u) << 4u) | ((px & 4u) << 4u) | ((px & 8u) >> 2u) | ((px & 16u) >> 2u) |
                 ((py & 1u) << 4u) | ((py & 2u) >> 1u) | ((py & 4u) << 5u) | ((py & 8u) << 5u);
        column ^= (((py >> 1u) ^ (py >> 2u)) & 1u) << 6u;
    } else if ((psm & 7u) == 2u) {
        bits = 16u; bx = px / 16u; by = py / 8u;
        if (psm & 8u) block = (by & 1u) | ((bx & 1u) << 1u) | (by & 4u) | ((by & 2u) << 2u) | ((bx & 2u) << 3u);
        else block = (by & 1u) | ((bx & 1u) << 1u) | ((by & 2u) << 1u) | ((bx & 2u) << 2u) | ((by & 4u) << 2u);
        column = ((px & 1u) << 1u) | ((px & 2u) << 2u) | ((px & 4u) << 2u) | ((px & 8u) >> 3u) |
                 ((py & 1u) << 2u) | ((py & 6u) << 4u);
    } else {
        bx = px / 8u; by = py / 8u;
        block = (bx & 1u) | ((by & 1u) << 1u) | ((bx & 2u) << 1u) | ((by & 2u) << 2u) | ((bx & 4u) << 2u);
        column = (px & 1u) | ((px & 6u) << 1u) | ((py & 1u) << 1u) | ((py & 6u) << 3u);
        if (psm == 27u || psm == 36u) high = 24u;
        if (psm == 44u) high = 28u;
    }
    block += bp + ((y >> sy) * ((bw * 64u) >> sx) + (x >> sx)) * 32u;
    if ((psm & 48u) == 48u) block ^= 24u;
    return (block * 2048u + column * bits + high) & 0x1FFFFFFu;
}

static inline uint32_t read_pixel(const uint32_t *memory, uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
{
    uint32_t a = gs_address(psm, bp, bw, x, y);
    return (load_word(memory, a >> 5) >> (a & 31u)) & pixel_mask(bits_per_pixel(psm));
}

/* Packed pixels may share a word with another thread's pixel: a word only partly written is
   changed with a compare-exchange; a whole word is stored. */
static inline void write_pixel(uint32_t *memory, uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y,
                               uint32_t value_in, uint32_t preserve)
{
    uint32_t a = gs_address(psm, bp, bw, x, y);
    uint32_t mask = (pixel_mask(bits_per_pixel(psm)) & ~preserve) << (a & 31u);
    uint32_t value = (value_in << (a & 31u)) & mask;
    uint32_t *word = &memory[a >> 5];
    if (mask == 0xFFFFFFFFu) { __atomic_store_n(word, value, __ATOMIC_RELAXED); return; }
    if (!mask) return;
    uint32_t old = __atomic_load_n(word, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(word, &old, (old & ~mask) | value, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

typedef struct { int32_t r, g, b, a; } Color;

static inline Color rgba(uint32_t v)
{
    Color c = {(int32_t)(v & 255u), (int32_t)((v >> 8) & 255u), (int32_t)((v >> 16) & 255u), (int32_t)(v >> 24)};
    return c;
}
static inline uint32_t pack32(Color v)
{
    return (uint32_t)v.r | ((uint32_t)v.g << 8) | ((uint32_t)v.b << 16) | ((uint32_t)v.a << 24);
}
static inline uint32_t pack16(uint32_t v)
{
    return ((v >> 3) & 31u) | ((v >> 6) & 0x3E0u) | ((v >> 9) & 0x7C00u) | ((v >> 16) & 0x8000u);
}
static inline int32_t clampi(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline Color clamp_color(Color c)
{
    Color r = {clampi(c.r, 0, 255), clampi(c.g, 0, 255), clampi(c.b, 0, 255), clampi(c.a, 0, 255)};
    return r;
}

/* The state's registers, as raster.wgsl reads them: regs[i].x the low word, .y the high one. */
#define REG_X(st, i) ((st)->reg[i][0])
#define REG_Y(st, i) ((st)->reg[i][1])

static inline uint32_t expand(const Constants *st, uint32_t value, uint32_t psm, int tex)
{
    if ((psm & 7u) == 2u) {
        uint32_t a = (value & 0x8000u) ? (REG_Y(st, 59) & 255u) : (REG_X(st, 59) & 255u);
        if (!tex) a = (value & 0x8000u) ? 128u : 0u;
        if (tex && (REG_X(st, 59) & 0x8000u) && value == 0u) a = 0u;
        return ((value & 31u) << 3) | ((value & 0x3E0u) << 6) | ((value & 0x7C00u) << 9) | (a << 24);
    }
    if ((psm & 7u) == 1u) {
        uint32_t a = tex ? (REG_X(st, 59) & 255u) : 128u;
        if (tex && (REG_X(st, 59) & 0x8000u) && value == 0u) a = 0u;
        return value | (a << 24);
    }
    return value;
}

static inline int32_t clamp_coordinate(int32_t v, uint32_t mode, uint32_t size, uint32_t low, uint32_t high)
{
    if (mode == 0u) return v & (int32_t)(size - 1u);
    if (mode == 1u) return clampi(v, 0, (int32_t)size - 1);
    if (mode == 2u) return clampi(v, (int32_t)low, (int32_t)high);
    return (v & (int32_t)low) | (int32_t)high;
}

static Color texel(const Constants *st, const uint32_t *texture, int32_t x, int32_t y, uint32_t ctxt, uint32_t level)
{
    uint32_t tx = REG_X(st, 6 + ctxt), ty = REG_Y(st, 6 + ctxt);
    uint32_t cx = REG_X(st, 8 + ctxt), cy = REG_Y(st, 8 + ctxt);
    uint32_t tw = (tx >> 26) & 15u, th = (tx >> 30) | ((ty & 3u) << 2);
    tw = tw > level ? tw - level : 0u;
    th = th > level ? th - level : 0u;
    x = clamp_coordinate(x, cx & 3u, 1u << tw, ((cx >> 4) & 1023u) >> level, ((cx >> 14) & 1023u) >> level);
    y = clamp_coordinate(y, (cx >> 2) & 3u, 1u << th, ((cx >> 24) | ((cy & 3u) << 8)) >> level, ((cy >> 2) & 1023u) >> level);
    uint32_t psm = (tx >> 20) & 63u, location = tx;
    if (level) {
        unsigned r = (level <= 3u ? 52u : 54u) + ctxt;
        uint32_t mx = REG_X(st, r), my = REG_Y(st, r), part = (level - 1u) % 3u;
        location = part == 0u ? mx : part == 1u ? (mx >> 20) | (my << 12) : my >> 8;
    }
    uint32_t v = read_pixel(texture, psm, location & 0x3FFFu, (location >> 14) & 63u, (uint32_t)x & 2047u, (uint32_t)y & 2047u);
    if (psm == 19u || psm == 20u || psm == 27u || psm == 36u || psm == 44u) v = st->palette[v & 255u];
    else v = expand(st, v, psm, 1);
    return rgba(v);
}

static Color sample_texture(const Constants *st, const uint32_t *texture, float sx, float sy, float q, uint32_t ctxt)
{
    uint32_t t1x = REG_X(st, 20 + ctxt), t1y = REG_Y(st, 20 + ctxt);
    uint32_t mmin = (t1x >> 6) & 7u, mxl = (t1x >> 2) & 7u;
    if (mxl > 6u) mxl = 6u;
    float k = (float)((int32_t)(t1y << 20) >> 20) / 16.0f;
    float lod = k;
    if (!(t1x & 1u)) {
        float scaled = log2f(fmaxf(fabsf(q), 1.17549435e-38f)) * (float)(1u << ((t1x >> 19) & 3u));
        lod = lod - scaled;
    }
    int filtered;
    if (mxl == 0u || lod <= 0.0f) filtered = (t1x & 32u) != 0u;
    else filtered = mmin == 1u || mmin == 4u || mmin == 5u;
    float clamped = lod < 0.0f ? 0.0f : lod > (float)mxl ? (float)mxl : lod;
    uint32_t fixed_lod = f2u(clamped * 65536.0f);
    uint32_t level = 0u, levels = 1u;
    if (mmin < 2u || mmin > 5u || mxl == 0u) {}
    else if (!(mmin & 1u)) level = (fixed_lod + 32768u) >> 16;
    else { level = fixed_lod >> 16; levels = 2u; }
    unsigned taps = filtered ? 4u : 1u;
    Color result[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    for (uint32_t i = 0; i < levels; i++) {
        uint32_t l = i == 1u ? (level + 1u < mxl ? level + 1u : mxl) : level;
        float scale = ldexpf(1.0f, -(int)l);
        float stx = sx * scale, sty = sy * scale;
        int32_t px, py, fx = 0, fy = 0;
        if (filtered) {
            float ax = stx - 0.5f, ay = sty - 0.5f, flx = floorf(ax), fly = floorf(ay);
            px = f2i(flx); py = f2i(fly);
            fx = f2i((ax - flx) * 16.0f); fy = f2i((ay - fly) * 16.0f);
        } else {
            px = f2i(floorf(stx)); py = f2i(floorf(sty));
        }
        Color sum = {0, 0, 0, 0};
        for (unsigned j = 0; j < taps; j++) {
            int32_t ox = (int32_t)(j & 1u), oy = (int32_t)(j >> 1);
            Color t = texel(st, texture, px + ox, py + oy, ctxt, l);
            int32_t w = (ox == 1 ? fx : 16 - fx) * (oy == 1 ? fy : 16 - fy);
            sum.r += t.r * w; sum.g += t.g * w; sum.b += t.b * w; sum.a += t.a * w;
        }
        result[i].r = sum.r >> 8; result[i].g = sum.g >> 8; result[i].b = sum.b >> 8; result[i].a = sum.a >> 8;
    }
    if (levels == 1u) return result[0];
    int32_t f = (int32_t)((fixed_lod & 65535u) >> 1);
    Color a = result[0], b = result[1], c;
    c.r = a.r + (((b.r - a.r) * f) >> 15); c.g = a.g + (((b.g - a.g) * f) >> 15);
    c.b = a.b + (((b.b - a.b) * f) >> 15); c.a = a.a + (((b.a - a.a) * f) >> 15);
    return c;
}

static inline float edge(int32_t ax, int32_t ay, int32_t bx, int32_t by, int32_t px, int32_t py)
{
    int32_t dx = bx - ax, dy = by - ay, qx = px - ax, qy = py - ay;
    uint32_t first = (uint32_t)abs(dx) * (uint32_t)abs(qy);
    uint32_t second = (uint32_t)abs(dy) * (uint32_t)abs(qx);
    int first_negative = (dx < 0) != (qy < 0);
    int second_negative = (dy < 0) != (qx < 0);
    float result;
    if (first_negative == second_negative) {
        if (first >= second) result = (float)(first - second);
        else result = -(float)(second - first);
    } else {
        result = (float)first + (float)second;
    }
    return first_negative ? -result : result;
}
static inline int top_left(int32_t ax, int32_t ay, int32_t bx, int32_t by)
{
    int32_t dx = bx - ax, dy = by - ay;
    return dy < 0 || (dy == 0 && dx > 0);
}
static inline int inside(float e, int32_t ax, int32_t ay, int32_t bx, int32_t by, int sign)
{
    int tl = sign > 0 ? top_left(ax, ay, bx, by) : top_left(bx, by, ax, ay);
    return e * (float)sign > 0.0f || (e == 0.0f && tl);
}

/* d0 + dzdx * (double)(float)dx + dzdy * (double)(float)dy, each step rounded (raster.wgsl's
   plane_depth, which emulates these doubles), then its integer part clamped to [0, 2^32 - 1]. */
static inline uint32_t plane_depth(const uint32_t words[8], int32_t dx, int32_t dy)
{
    double d[3];
    memcpy(d, words, sizeof d);
    double z = d[0];
    double term = d[1] * (double)(float)dx;
    z = z + term;
    term = d[2] * (double)(float)dy;
    z = z + term;
    if (!(z >= 1.0)) return 0u;
    if (z >= 4294967296.0) return 0xFFFFFFFFu;
    return (uint32_t)z;
}

/* What draw_pixel needs of a triangle that is the same at every pixel: its doubled area's edge
   value and its sign (draw_tile works them out once per tile). */
typedef struct { float area; int sign; } Setup;

/* raster.wgsl's draw_pixel: primitive d (of state st) at pixel (x, y). */
static void draw_pixel(uint32_t *memory, const uint32_t *texture, const Primitive *d, const Constants *st,
                       const Setup *setup, uint32_t x, uint32_t y)
{
    uint32_t scanmask = REG_X(st, 34) & 3u;
    if ((scanmask == 2u && !(y & 1u)) || (scanmask == 3u && (y & 1u))) return;
    uint32_t prim = d->info[1], ctxt = (prim >> 9) & 1u, type = d->info[0], last = d->info[2] - 1u;
    int32_t px = (int32_t)x * 16, py = (int32_t)y * 16;
    int32_t ax = (int32_t)d->position[0][0], ay = (int32_t)d->position[0][1];
    int32_t bx = (int32_t)d->position[1][0], by = (int32_t)d->position[1][1];
    int32_t cx = (int32_t)d->position[2][0], cy = (int32_t)d->position[2][1];
    float w0 = 1.0f, w1 = 0.0f, w2 = 0.0f, fx = 0.0f, fy = 0.0f;
    if (type == 0u) {
        if ((ax >> 4) != (int32_t)x || (ay >> 4) != (int32_t)y) return;
    } else if (type == 6u) {
        int32_t lx = ax < bx ? ax : bx, hx = ax < bx ? bx : ax, ly = ay < by ? ay : by, hy = ay < by ? by : ay;
        if (px < lx || py < ly || px >= hx || py >= hy) return;
        fx = (float)(px - ax) / (float)(bx - ax);
        fy = (float)(py - ay) / (float)(by - ay);
        w0 = 0.0f; w1 = 1.0f;
    } else if (type == 1u || type == 2u) {
        int32_t ddx = bx - ax, ddy = by - ay;
        int32_t major = abs(ddx) > abs(ddy) ? abs(ddx) : abs(ddy);
        if (major == 0) return;
        float along;
        if (abs(ddx) >= abs(ddy)) along = (float)(px - ax) / (float)ddx;
        else along = (float)(py - ay) / (float)ddy;
        if (along < 0.0f || along >= 1.0f || fabsf(edge(ax, ay, bx, by, px, py)) >= (float)major * 8.0f) return;
        w0 = 1.0f - along; w1 = along;
    } else {
        /* the vertices' box and a zero area are draw_tile's: it visits only pixels inside the box */
        float area = setup->area;
        int sign = setup->sign;
        float e0 = edge(bx, by, cx, cy, px, py), e1 = edge(cx, cy, ax, ay, px, py), e2 = edge(ax, ay, bx, by, px, py);
        if (!inside(e0, bx, by, cx, cy, sign) || !inside(e1, cx, cy, ax, ay, sign) || !inside(e2, ax, ay, bx, by, sign)) return;
        w0 = e0 / area; w1 = e1 / area; w2 = e2 / area;
    }
    Color color = rgba(d->position[last][3]);
    if ((prim & 8u) && type != 6u) {
        Color c0 = rgba(d->position[0][3]), c1 = rgba(d->position[1][3]), c2 = rgba(d->position[2][3]);
        float r0 = (float)c0.r * w0, r1 = (float)c1.r * w1, r2 = (float)c2.r * w2;
        float g0 = (float)c0.g * w0, g1 = (float)c1.g * w1, g2 = (float)c2.g * w2;
        float b0 = (float)c0.b * w0, b1 = (float)c1.b * w1, b2 = (float)c2.b * w2;
        float a0 = (float)c0.a * w0, a1 = (float)c1.a * w1, a2 = (float)c2.a * w2;
        float r = r0 + r1, g = g0 + g1, b = b0 + b1, a = a0 + a1;
        color.r = f2i(r + r2); color.g = f2i(g + g2); color.b = f2i(b + b2); color.a = f2i(a + a2);
    }
    uint32_t z = d->position[last][2];
    int constant_z = d->position[0][2] == d->position[1][2] && (d->info[2] < 3u || d->position[0][2] == d->position[2][2]);
    if (type != 0u && type != 6u && !constant_z) z = plane_depth(d->depth, px - ax, py - ay);
    /* the host's independent-pixels proof allows the early depth test (raster.wgsl EARLY_DEPTH) */
    int early_depth = (st->info[1] & 1u) != 0u;
    if (early_depth) {
        uint32_t early_test = REG_X(st, 71 + ctxt);
        if (early_test & 0x10000u) {
            uint32_t early_frame = REG_X(st, 76 + ctxt), early_zbuf = REG_X(st, 78 + ctxt);
            uint32_t early_zpsm = ((early_zbuf >> 24) & 15u) | 48u;
            uint32_t limit = pixel_mask(bits_per_pixel(early_zpsm));
            uint32_t early_z = z < limit ? z : limit;
            uint32_t old_z = read_pixel(memory, early_zpsm, (early_zbuf & 511u) * 32u, (early_frame >> 16) & 63u, x, y);
            uint32_t ztest = (early_test >> 17) & 3u;
            if (ztest == 0u || (ztest == 2u && early_z < old_z) || (ztest == 3u && early_z <= old_z)) return;
        }
    }
    if (prim & 16u) {
        float ix, iy, iz, sx, sy;
        {
            float x0 = d->stq[0][0] * w0, x1 = d->stq[1][0] * w1, x2 = d->stq[2][0] * w2;
            float y0 = d->stq[0][1] * w0, y1 = d->stq[1][1] * w1, y2 = d->stq[2][1] * w2;
            float z0 = d->stq[0][2] * w0, z1 = d->stq[1][2] * w1, z2 = d->stq[2][2] * w2;
            ix = (x0 + x1) + x2; iy = (y0 + y1) + y2; iz = (z0 + z1) + z2;
        }
        if (type == 6u) {
            float ox = (d->stq[1][0] - d->stq[0][0]) * fx, oy = (d->stq[1][1] - d->stq[0][1]) * fy;
            ix = d->stq[0][0] + ox; iy = d->stq[0][1] + oy; iz = d->stq[1][2];
        }
        if (prim & 256u) {
            float u0 = (float)d->uv[0][0], v0 = (float)d->uv[0][1], u1 = (float)d->uv[1][0], v1 = (float)d->uv[1][1];
            float u2 = (float)d->uv[2][0], v2 = (float)d->uv[2][1];
            if (type == 6u) {
                float ou = (u1 - u0) * fx, ov = (v1 - v0) * fy;
                sx = (u0 + ou) / 16.0f; sy = (v0 + ov) / 16.0f;
            } else {
                float a0 = u0 * w0, a1 = u1 * w1, a2 = u2 * w2, b0 = v0 * w0, b1 = v1 * w1, b2 = v2 * w2;
                sx = ((a0 + a1) + a2) / 16.0f; sy = ((b0 + b1) + b2) / 16.0f;
            }
        } else {
            uint32_t tx = REG_X(st, 6 + ctxt), ty = REG_Y(st, 6 + ctxt);
            float width = (float)(1u << ((tx >> 26) & 15u)), height = (float)(1u << ((tx >> 30) | ((ty & 3u) << 2)));
            float q = iz == 0.0f ? 1.17549435e-38f : iz;
            float mx = ix * width, my = iy * height;
            sx = mx / q; sy = my / q;
        }
        Color tex = sample_texture(st, texture, sx, sy, iz, ctxt);
        uint32_t t0y = REG_Y(st, 6 + ctxt);
        uint32_t tfx = (t0y >> 3) & 3u, tcc = (t0y >> 2) & 1u;
        int32_t alpha = color.a;
        if (tfx == 0u) {
            color.r = (color.r * tex.r) >> 7; color.g = (color.g * tex.g) >> 7;
            color.b = (color.b * tex.b) >> 7; color.a = (color.a * tex.a) >> 7;
        } else if (tfx == 1u) {
            color = tex;
        } else {
            color.r = ((color.r * tex.r) >> 7) + alpha; color.g = ((color.g * tex.g) >> 7) + alpha;
            color.b = ((color.b * tex.b) >> 7) + alpha;
            color.a = tfx == 2u ? tex.a + alpha : tex.a;
        }
        if (tcc == 0u) color.a = alpha;
        color = clamp_color(color);
    }
    if (prim & 32u) {
        float f0 = (float)d->uv[0][2] * w0, f1 = (float)d->uv[1][2] * w1, f2 = (float)d->uv[2][2] * w2;
        int32_t fog = (int32_t)f2u((f0 + f1) + f2);
        Color fc = rgba(REG_X(st, 61));
        color.r = fc.r + (((color.r - fc.r) * fog) >> 8);
        color.g = fc.g + (((color.g - fc.g) * fog) >> 8);
        color.b = fc.b + (((color.b - fc.b) * fog) >> 8);
    }
    uint32_t test = REG_X(st, 71 + ctxt);
    int write_frame = 1, write_z = (REG_Y(st, 78 + ctxt) & 1u) == 0u, rgb_only = 0;
    if (test & 1u) {
        uint32_t atst = (test >> 1) & 7u;
        int32_t ref = (int32_t)((test >> 4) & 255u);
        int pass = atst == 1u || (atst == 2u && color.a < ref) || (atst == 3u && color.a <= ref) ||
                   (atst == 4u && color.a == ref) || (atst == 5u && color.a >= ref) ||
                   (atst == 6u && color.a > ref) || (atst == 7u && color.a != ref);
        if (!pass) {
            uint32_t afail = (test >> 12) & 3u;
            if (afail == 0u) return;
            write_frame = afail != 2u; write_z = write_z && afail == 2u; rgb_only = afail == 3u;
        }
    }
    uint32_t frame = REG_X(st, 76 + ctxt), bw = (frame >> 16) & 63u, bp = (frame & 511u) * 32u, psm = (frame >> 24) & 63u;
    uint32_t zbuf = REG_X(st, 78 + ctxt), zpsm = ((zbuf >> 24) & 15u) | 48u, zbp = (zbuf & 511u) * 32u;
    uint32_t zlimit = pixel_mask(bits_per_pixel(zpsm));
    if (z > zlimit) z = zlimit;
    uint32_t raw = read_pixel(memory, psm, bp, bw, x, y);
    Color dst = rgba(expand(st, raw, psm, 0));
    if ((test & 0x4000u) && psm != 1u && ((dst.a >> 7) != (int32_t)((test >> 15) & 1u))) return;
    if (test & 0x10000u) {
        if (!early_depth) {
            uint32_t ztest = (test >> 17) & 3u, old_z = read_pixel(memory, zpsm, zbp, bw, x, y);
            if (ztest == 0u || (ztest == 2u && z < old_z) || (ztest == 3u && z <= old_z)) return;
        }
    } else {
        write_z = 0;
    }
    if (write_frame) {
        if ((prim & 64u) && (!(REG_X(st, 73) & 1u) || (color.a & 128))) {
            uint32_t alpha = REG_X(st, 66 + ctxt);
            uint32_t sa = alpha & 3u, sb = (alpha >> 2) & 3u, sc = (alpha >> 4) & 3u, sd = (alpha >> 6) & 3u;
            Color zero = {0, 0, 0, 0};
            Color av = sa == 0u ? color : sa == 1u ? dst : zero;
            Color bv = sb == 0u ? color : sb == 1u ? dst : zero;
            int32_t cv = sc == 0u ? color.a : sc == 1u ? dst.a : (int32_t)(REG_Y(st, 66 + ctxt) & 255u);
            Color dv = sd == 0u ? color : sd == 1u ? dst : zero;
            color.r = (((av.r - bv.r) * cv) >> 7) + dv.r;
            color.g = (((av.g - bv.g) * cv) >> 7) + dv.g;
            color.b = (((av.b - bv.b) * cv) >> 7) + dv.b;
        }
        if ((psm & 7u) == 2u && (REG_X(st, 69) & 1u)) {
            uint32_t index = (y & 3u) * 4u + (x & 3u);
            uint32_t word = st->reg[68][index >> 3];
            int32_t dither = (int32_t)(((word >> ((index & 7u) * 4u)) & 7u) << 29) >> 29;
            color.r += dither; color.g += dither; color.b += dither;
        }
        if (REG_X(st, 70) & 1u) color = clamp_color(color);
        else { color.r &= 255; color.g &= 255; color.b &= 255; color.a &= 255; }
        if (REG_X(st, 74 + ctxt) & 1u) color.a |= 128;
        uint32_t mask = REG_Y(st, 76 + ctxt) | (rgb_only ? 0xFF000000u : 0u);
        uint32_t value = pack32(color);
        if ((psm & 7u) == 2u) { value = pack16(value); mask = pack16(mask); }
        write_pixel(memory, psm, bp, bw, x, y, value, mask);
    }
    if (write_z) write_pixel(memory, zpsm, zbp, bw, x, y, z, 0u);
}

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
        uint32_t left = x0 > d->bounds[0] ? x0 : d->bounds[0];
        uint32_t top = y0 > d->bounds[1] ? y0 : d->bounds[1];
        uint32_t right = d->bounds[0] + d->bounds[2], bottom = d->bounds[1] + d->bounds[3];
        if (right > x0 + 8u) right = x0 + 8u;
        if (bottom > y0 + 8u) bottom = y0 + 8u;
        Setup setup = {0.0f, 1};
        int32_t ax = (int32_t)d->position[0][0], ay = (int32_t)d->position[0][1];
        int32_t bx = (int32_t)d->position[1][0], by = (int32_t)d->position[1][1];
        int32_t box[4] = {(int32_t)left, (int32_t)top, (int32_t)right, (int32_t)bottom};
        uint32_t type = d->info[0];
        if (type == 0u) {
            /* a point: its one pixel */
            box[0] = ax >> 4; box[1] = ay >> 4; box[2] = box[0] + 1; box[3] = box[1] + 1;
        } else if (type == 6u) {
            /* a sprite covers the pixels whose centres p * 16 lie in [min, max) */
            int32_t lx = ax < bx ? ax : bx, hx = ax < bx ? bx : ax, ly = ay < by ? ay : by, hy = ay < by ? by : ay;
            box[0] = (lx + 15) >> 4; box[1] = (ly + 15) >> 4; box[2] = (hx + 15) >> 4; box[3] = (hy + 15) >> 4;
        } else if (type != 1u && type != 2u) {
            /* a triangle: the pixels in [min, max] of its vertices, and nothing at a zero area */
            int32_t cx = (int32_t)d->position[2][0], cy = (int32_t)d->position[2][1];
            int32_t lx = ax, hx = ax, ly = ay, hy = ay;
            if (bx < lx) lx = bx; if (cx < lx) lx = cx; if (bx > hx) hx = bx; if (cx > hx) hx = cx;
            if (by < ly) ly = by; if (cy < ly) ly = cy; if (by > hy) hy = by; if (cy > hy) hy = cy;
            setup.area = edge(ax, ay, bx, by, cx, cy);
            if (setup.area == 0.0f) continue;
            setup.sign = setup.area > 0.0f ? 1 : -1;
            box[0] = (lx + 15) >> 4; box[1] = (ly + 15) >> 4; box[2] = (hx >> 4) + 1; box[3] = (hy >> 4) + 1;
        }
        if (type != 1u && type != 2u) {
            if (box[0] < (int32_t)left) box[0] = (int32_t)left;
            if (box[1] < (int32_t)top) box[1] = (int32_t)top;
            if (box[2] > (int32_t)right) box[2] = (int32_t)right;
            if (box[3] > (int32_t)bottom) box[3] = (int32_t)bottom;
        }
        for (int32_t y = box[1]; y < box[3]; y++)
            for (int32_t x = box[0]; x < box[2]; x++)
                draw_pixel(cpu->vram, cpu->snapshot, d, st, &setup, (uint32_t)x, (uint32_t)y);
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
static uint32_t scan_pixel(Cpu *cpu, const uint32_t c[12], uint32_t x, uint32_t y)
{
    uint32_t v = read_pixel(cpu->vram, c[0], c[1], c[2], (c[3] + x) & 2047u, (c[4] + y) & 2047u);
    if (c[0] == 2u || c[0] == 10u) v = ((v & 31u) << 3) | ((v & 0x3E0u) << 6) | ((v & 0x7C00u) << 9);
    return (v & 0xFFFFFFu) | 0xFF000000u;
}

/* raster.wgsl's scanout: the display circuit's picture (VRAM itself) into the screen image.

   Field mode (c[7]: interlaced, one field's lines in the frame buffer, each shown twice): the game
   draws each field half a line apart, for a TV's interlace, which shows the fields' lines between
   each other. A progressive screen shows that as the picture jumping a line up and down every
   field. So a field is shown one line lower when that matches the picture before it better
   (sampled differences of the green channel; a tie keeps the last choice), which keeps it still. */
static void run_scanout(Cpu *cpu, const uint32_t c[12], uint32_t width, uint32_t height)
{
    static int shift;  /* the last field's: 0, or 1 = shown one line lower */
    if (!c[7]) {
        for (uint32_t y = 0; y < height; y++) {
            uint32_t *row = cpu->screen + (size_t)y * width;
            for (uint32_t x = 0; x < width; x++) row[x] = scan_pixel(cpu, c, x, y);
        }
        cpu->screen_width = width;
        cpu->screen_height = height;
        return;
    }
    uint32_t lines = (height + 1) / 2;
    for (uint32_t y = 0; y < lines; y++) {
        uint32_t *row = cpu->field + (size_t)y * width;
        for (uint32_t x = 0; x < width; x++) row[x] = scan_pixel(cpu, c, x, y);
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

static void op_submit(Exact *gpu)
{
    Cpu *cpu = (Cpu *)gpu;
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
    cpu->vram = calloc(1, GS_VRAM_SIZE);
    cpu->snapshot = calloc(1, GS_VRAM_SIZE);
    cpu->readback = calloc(1, GS_VRAM_SIZE);
    cpu->staged = calloc(1, GS_VRAM_SIZE);
    cpu->upload = calloc(1, GS_VRAM_SIZE);
    cpu->screen = calloc(1024 * 1024, 4);
    cpu->field = calloc(1024 * 1024, 4);
    if (!cpu->vram || !cpu->snapshot || !cpu->readback || !cpu->staged || !cpu->upload || !cpu->screen || !cpu->field ||
        !exact_alloc(&cpu->e)) {
        fprintf(stderr, "gs: out of memory for the CPU renderer\n");
        gs_gpu_destroy(g);
        return 0;
    }
    const char *limit = getenv("DHD_GS_BATCH");
    exact_config(&cpu->e, getenv("DHD_PROFILE") != NULL, limit ? strtoul(limit, NULL, 0) : 0);
    exact_hooks(g, &cpu->e);
    if (!pool.workers) pool_start();
    gnm_init();  /* the GPU path's check (port/ps4/gnm.c): its result goes to the log */
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
    exact_free(&cpu->e);
    free(cpu->vram); free(cpu->snapshot); free(cpu->readback); free(cpu->staged); free(cpu->upload);
    free(cpu->screen); free(cpu->field); free(cpu->commands);
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
        fprintf(stderr, "gs cpu: %.1f fields/s, %.0f draws/s, %.0f batches/s, %.0f tiles/s, raster %.0f ms/s, "
                "%.1f snapshots/s (%.0f pages/s), %.1f uploads/s, %.1f local copies/s\n",
                fields / seconds, cpu->e.draws / seconds, cpu->batches / seconds, cpu->tiles / seconds,
                cpu->raster_ticks / 1000.0 / seconds, cpu->snapshots / seconds, cpu->snapshot_pages / seconds,
                cpu->e.uploads / seconds, cpu->e.copies / seconds);
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
