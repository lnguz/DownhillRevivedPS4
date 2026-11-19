/* raster.wgsl's arithmetic in C, shared by the PS4's two GS devices (port/ps4/PLAN.md): the CPU's
   (gpu_cpu.c, worker threads) and the GPU's (port/ps4/kernels/raster.cl, OpenCL C that LLVM builds
   for the console's GCN GPU). One copy of the rules, so both draw the same pixels.

   The translation keeps the shader's arithmetic: float math in the same order (no FMA: FP_CONTRACT
   off on both), D3D's float-to-integer rules, and the depth plane in double precision, which the
   shader emulates (sd_*) and both devices have. Packed pixels may share a word with another
   thread's pixel: a word only partly written is changed with a compare-exchange.

   The includer provides: Primitive and Constants (gpu_exact.inc's, the shader's Primitive and
   State), RC_MEM (the qualifier of device memory pointers), RC_LOAD, RC_STORE and RC_CAS (a
   compare-exchange that returns the word it found), rc_abs, and the fixed-width integer types. */
#ifndef PORT_PS4_RASTER_CORE_H
#define PORT_PS4_RASTER_CORE_H

/* RC_INLINE: the GPU build inlines the pixel's work into the kernel's loop (raster.cl): as a call,
   the function ABI held 145 VGPRs (one wave per SIMD); inlined it takes 90 (two). */
#ifndef RC_INLINE
#define RC_INLINE
#endif

static inline uint32_t load_word(RC_MEM const uint32_t *memory, uint32_t word) { return RC_LOAD(&memory[word]); }

static inline uint32_t float_bits(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    return v.u;
}

/* D3D's ftoi/ftou: NaN gives 0 and out-of-range values saturate. */
static inline int32_t f2i(float x)
{
    uint32_t bits = float_bits(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0;
    if (x >= 2147483648.0f) return 2147483647;
    if (x <= -2147483648.0f) return (int32_t)(-2147483647 - 1);
    return (int32_t)x;
}
static inline uint32_t f2u(float x)
{
    uint32_t bits = float_bits(x);
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

static inline uint32_t read_pixel(RC_MEM const uint32_t *memory, uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
{
    uint32_t a = gs_address(psm, bp, bw, x, y);
    return (load_word(memory, a >> 5) >> (a & 31u)) & pixel_mask(bits_per_pixel(psm));
}

/* Packed pixels may share a word with another thread's pixel: a word only partly written is
   changed with a compare-exchange; a whole word is stored. */
static inline void write_pixel(RC_MEM uint32_t *memory, uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y,
                               uint32_t value_in, uint32_t preserve)
{
    uint32_t a = gs_address(psm, bp, bw, x, y);
    uint32_t mask = (pixel_mask(bits_per_pixel(psm)) & ~preserve) << (a & 31u);
    uint32_t value = (value_in << (a & 31u)) & mask;
    RC_MEM uint32_t *word = &memory[a >> 5];
    if (mask == 0xFFFFFFFFu) { RC_STORE(word, value); return; }
    if (!mask) return;
    uint32_t old = RC_LOAD(word);
    for (;;) {
        uint32_t seen = RC_CAS(word, old, (old & ~mask) | value);
        if (seen == old) break;
        old = seen;
    }
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

static inline uint32_t expand(RC_MEM const Constants *st, uint32_t value, uint32_t psm, int tex)
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

static RC_INLINE Color texel(RC_MEM const Constants *st, RC_MEM const uint32_t *texture, int32_t x, int32_t y, uint32_t ctxt, uint32_t level)
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

static RC_INLINE Color sample_texture(RC_MEM const Constants *st, RC_MEM const uint32_t *texture, float sx, float sy, float q, uint32_t ctxt)
{
    uint32_t t1x = REG_X(st, 20 + ctxt), t1y = REG_Y(st, 20 + ctxt);
    uint32_t mmin = (t1x >> 6) & 7u, mxl = (t1x >> 2) & 7u;
    if (mxl > 6u) mxl = 6u;
    float k = (float)((int32_t)(t1y << 20) >> 20) / 16.0f;
    float lod = k;
    if (!(t1x & 1u)) {
        float scaled = __builtin_log2f(__builtin_fmaxf(__builtin_fabsf(q), 1.17549435e-38f)) * (float)(1u << ((t1x >> 19) & 3u));
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
    Color first = {0, 0, 0, 0}, second = {0, 0, 0, 0};
    for (uint32_t i = 0; i < levels; i++) {
        uint32_t l = i == 1u ? (level + 1u < mxl ? level + 1u : mxl) : level;
        float scale = __builtin_ldexpf(1.0f, -(int)l);
        float stx = sx * scale, sty = sy * scale;
        int32_t px, py, fx = 0, fy = 0;
        if (filtered) {
            float ax = stx - 0.5f, ay = sty - 0.5f, flx = __builtin_floorf(ax), fly = __builtin_floorf(ay);
            px = f2i(flx); py = f2i(fly);
            fx = f2i((ax - flx) * 16.0f); fy = f2i((ay - fly) * 16.0f);
        } else {
            px = f2i(__builtin_floorf(stx)); py = f2i(__builtin_floorf(sty));
        }
        /* The taps written out, not looped over: their memory reads (and the palette's) are
           independent and go out together; in a loop each waited for the one before (a GPU
           thread idled through eight reads in a row per filtered pixel). The sum is the loop's,
           term by term in the same order (integers). */
        Color sum = {0, 0, 0, 0};
        if (taps == 4u) {
            Color t0 = texel(st, texture, px, py, ctxt, l), t1 = texel(st, texture, px + 1, py, ctxt, l);
            Color t2 = texel(st, texture, px, py + 1, ctxt, l), t3 = texel(st, texture, px + 1, py + 1, ctxt, l);
            int32_t w0 = (16 - fx) * (16 - fy), w1 = fx * (16 - fy), w2 = (16 - fx) * fy, w3 = fx * fy;
            sum.r = t0.r * w0 + t1.r * w1 + t2.r * w2 + t3.r * w3;
            sum.g = t0.g * w0 + t1.g * w1 + t2.g * w2 + t3.g * w3;
            sum.b = t0.b * w0 + t1.b * w1 + t2.b * w2 + t3.b * w3;
            sum.a = t0.a * w0 + t1.a * w1 + t2.a * w2 + t3.a * w3;
        } else {
            Color t0 = texel(st, texture, px, py, ctxt, l);
            int32_t w0 = (16 - fx) * (16 - fy);
            sum.r = t0.r * w0; sum.g = t0.g * w0; sum.b = t0.b * w0; sum.a = t0.a * w0;
        }
        sum.r >>= 8; sum.g >>= 8; sum.b >>= 8; sum.a >>= 8;
        if (i == 0u) first = sum; else second = sum;
    }
    if (levels == 1u) return first;
    int32_t f = (int32_t)((fixed_lod & 65535u) >> 1);
    Color a = first, b = second, c;
    c.r = a.r + (((b.r - a.r) * f) >> 15); c.g = a.g + (((b.g - a.g) * f) >> 15);
    c.b = a.b + (((b.b - a.b) * f) >> 15); c.a = a.a + (((b.a - a.a) * f) >> 15);
    return c;
}

static inline float edge(int32_t ax, int32_t ay, int32_t bx, int32_t by, int32_t px, int32_t py)
{
    int32_t dx = bx - ax, dy = by - ay, qx = px - ax, qy = py - ay;
    uint32_t first = (uint32_t)rc_abs(dx) * (uint32_t)rc_abs(qy);
    uint32_t second = (uint32_t)rc_abs(dy) * (uint32_t)rc_abs(qx);
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
static inline double word_double(uint32_t lo, uint32_t hi)
{
    union { uint64_t u; double d; } v;
    v.u = (uint64_t)lo | (uint64_t)hi << 32;
    return v.d;
}

static inline uint32_t plane_depth(RC_MEM const uint32_t *words, int32_t dx, int32_t dy)
{
    double z = word_double(words[0], words[1]);
    double term = word_double(words[2], words[3]) * (double)(float)dx;
    z = z + term;
    term = word_double(words[4], words[5]) * (double)(float)dy;
    z = z + term;
    if (!(z >= 1.0)) return 0u;
    if (z >= 4294967296.0) return 0xFFFFFFFFu;
    return (uint32_t)z;
}

/* A pixel's frame and Z words, kept by the thread that draws it (raster.cl: one thread per pixel
   for a whole batch). With the host's independent-pixels proof (State.info.y bit 0) no other
   thread writes this pixel during the batch, so after the first read the pixel's bits come from
   here: a tile under hundreds of layers (the menu's distant geometry) waited on a memory read per
   layer. A word holding one pixel (32 and 24 bits: PSMCT32/24, PSMZ32/24) is then stored without
   a compare-exchange, as raster.wgsl's EXCLUSIVE_WORD_STORE does; packed words still use one.
   address 0: empty. Without the proof nothing is kept. */
typedef struct { uint32_t address, word; } WordCache;
typedef struct { WordCache frame, z; } PixelCache;

static inline uint32_t read_kept(RC_MEM const uint32_t *memory, WordCache *keep, int keeping, uint32_t psm,
                                 uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
{
    uint32_t a = gs_address(psm, bp, bw, x, y), w = a >> 5;
    uint32_t word;
    if (keeping && keep->address == w + 1u) word = keep->word;
    else {
        word = load_word(memory, w);
        if (keeping) { keep->address = w + 1u; keep->word = word; }
    }
    return (word >> (a & 31u)) & pixel_mask(bits_per_pixel(psm));
}

static inline void write_kept(RC_MEM uint32_t *memory, WordCache *keep, int keeping, uint32_t psm, uint32_t bp,
                              uint32_t bw, uint32_t x, uint32_t y, uint32_t value_in, uint32_t preserve)
{
    if (!keeping) { write_pixel(memory, psm, bp, bw, x, y, value_in, preserve); return; }
    uint32_t a = gs_address(psm, bp, bw, x, y), w = a >> 5;
    uint32_t mask = (pixel_mask(bits_per_pixel(psm)) & ~preserve) << (a & 31u);
    uint32_t value = (value_in << (a & 31u)) & mask;
    if (!mask) return;
    RC_MEM uint32_t *word = &memory[w];
    uint32_t old = keep->address == w + 1u ? keep->word : (mask == 0xFFFFFFFFu ? 0u : RC_LOAD(word));
    uint32_t updated = (old & ~mask) | value;
    if (mask == 0xFFFFFFFFu || psm == 0u || psm == 1u || psm == 48u || psm == 49u) {
        RC_STORE(word, updated);  /* the word is this pixel's alone */
    } else {
        for (;;) {
            uint32_t seen = RC_CAS(word, old, updated);
            if (seen == old) break;
            old = seen;
            updated = (old & ~mask) | value;
        }
    }
    keep->address = w + 1u;
    keep->word = updated;
}

/* What draw_pixel needs of a triangle that is the same at every pixel: its doubled area's edge
   value and its sign (draw_tile works them out once per tile). */
typedef struct { float area; int sign; } Setup;

/* raster.wgsl's draw_pixel: primitive d (of state st) at pixel (x, y). */
static RC_INLINE void draw_pixel(RC_MEM uint32_t *memory, RC_MEM const uint32_t *texture, RC_MEM const Primitive *d, RC_MEM const Constants *st,
                       const Setup *setup, PixelCache *cache, uint32_t x, uint32_t y)
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
        int32_t major = rc_abs(ddx) > rc_abs(ddy) ? rc_abs(ddx) : rc_abs(ddy);
        if (major == 0) return;
        float along;
        if (rc_abs(ddx) >= rc_abs(ddy)) along = (float)(px - ax) / (float)ddx;
        else along = (float)(py - ay) / (float)ddy;
        if (along < 0.0f || along >= 1.0f || __builtin_fabsf(edge(ax, ay, bx, by, px, py)) >= (float)major * 8.0f) return;
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
            uint32_t old_z = read_kept(memory, &cache->z, 1, early_zpsm, (early_zbuf & 511u) * 32u, (early_frame >> 16) & 63u, x, y);
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
    uint32_t raw = read_kept(memory, &cache->frame, early_depth, psm, bp, bw, x, y);
    Color dst = rgba(expand(st, raw, psm, 0));
    if ((test & 0x4000u) && psm != 1u && ((dst.a >> 7) != (int32_t)((test >> 15) & 1u))) return;
    if (test & 0x10000u) {
        if (!early_depth) {
            uint32_t ztest = (test >> 17) & 3u, old_z = read_kept(memory, &cache->z, early_depth, zpsm, zbp, bw, x, y);
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
        write_kept(memory, &cache->frame, early_depth, psm, bp, bw, x, y, value, mask);
    }
    if (write_z) write_kept(memory, &cache->z, early_depth, zpsm, zbp, bw, x, y, z, 0u);
}

/* A primitive's pixels inside the 8x8 tile at (x0, y0): box (left, top, right, bottom, the last two
   past the end) and the triangle setup. 0 when it has none there (a zero area). Lines keep the
   tile's box: their own test is draw_pixel's. */
static inline int prim_box(RC_MEM const Primitive *d, uint32_t x0, uint32_t y0, Setup *setup, int32_t *box)
{
    uint32_t left = x0 > d->bounds[0] ? x0 : d->bounds[0];
    uint32_t top = y0 > d->bounds[1] ? y0 : d->bounds[1];
    uint32_t right = d->bounds[0] + d->bounds[2], bottom = d->bounds[1] + d->bounds[3];
    if (right > x0 + 8u) right = x0 + 8u;
    if (bottom > y0 + 8u) bottom = y0 + 8u;
    setup->area = 0.0f;
    setup->sign = 1;
    int32_t ax = (int32_t)d->position[0][0], ay = (int32_t)d->position[0][1];
    int32_t bx = (int32_t)d->position[1][0], by = (int32_t)d->position[1][1];
    box[0] = (int32_t)left; box[1] = (int32_t)top; box[2] = (int32_t)right; box[3] = (int32_t)bottom;
    uint32_t type = d->info[0];
    int32_t b0, b1, b2, b3;
    if (type == 1u || type == 2u) {
        /* a line: draw_pixel's test (along the major axis in [0, 1), within half a pixel across)
           leaves nothing outside its endpoints' pixels widened by one */
        int32_t lx = ax < bx ? ax : bx, hx = ax < bx ? bx : ax, ly = ay < by ? ay : by, hy = ay < by ? by : ay;
        b0 = (lx >> 4) - 1; b1 = (ly >> 4) - 1; b2 = (hx >> 4) + 2; b3 = (hy >> 4) + 2;
    } else if (type == 0u) {
        /* a point: its one pixel */
        b0 = ax >> 4; b1 = ay >> 4; b2 = b0 + 1; b3 = b1 + 1;
    } else if (type == 6u) {
        /* a sprite covers the pixels whose centres p * 16 lie in [min, max) */
        int32_t lx = ax < bx ? ax : bx, hx = ax < bx ? bx : ax, ly = ay < by ? ay : by, hy = ay < by ? by : ay;
        b0 = (lx + 15) >> 4; b1 = (ly + 15) >> 4; b2 = (hx + 15) >> 4; b3 = (hy + 15) >> 4;
    } else {
        /* a triangle: the pixels in [min, max] of its vertices, and nothing at a zero area */
        int32_t cx = (int32_t)d->position[2][0], cy = (int32_t)d->position[2][1];
        int32_t lx = ax, hx = ax, ly = ay, hy = ay;
        if (bx < lx) lx = bx;
        if (cx < lx) lx = cx;
        if (bx > hx) hx = bx;
        if (cx > hx) hx = cx;
        if (by < ly) ly = by;
        if (cy < ly) ly = cy;
        if (by > hy) hy = by;
        if (cy > hy) hy = cy;
        setup->area = edge(ax, ay, bx, by, cx, cy);
        if (setup->area == 0.0f) return 0;
        setup->sign = setup->area > 0.0f ? 1 : -1;
        b0 = (lx + 15) >> 4; b1 = (ly + 15) >> 4; b2 = (hx >> 4) + 1; b3 = (hy >> 4) + 1;
    }
    if (b0 > box[0]) box[0] = b0;
    if (b1 > box[1]) box[1] = b1;
    if (b2 < box[2]) box[2] = b2;
    if (b3 < box[3]) box[3] = b3;
    return box[0] < box[2] && box[1] < box[3];
}

/* The display picture's pixel (x, y) of VRAM as RGBA with alpha 255 (raster.wgsl's scanout). */
static inline uint32_t scan_pixel(RC_MEM const uint32_t *vram, const uint32_t *c, uint32_t x, uint32_t y)
{
    uint32_t v = read_pixel(vram, c[0], c[1], c[2], (c[3] + x) & 2047u, (c[4] + y) & 2047u);
    if (c[0] == 2u || c[0] == 10u) v = ((v & 31u) << 3) | ((v & 0x3E0u) << 6) | ((v & 0x7C00u) << 9);
    return (v & 0xFFFFFFu) | 0xFF000000u;
}
#endif
