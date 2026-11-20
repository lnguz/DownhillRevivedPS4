/* The native renderer's vertex and pixel stages on the PS4 (port/ps4/render_ps4.c): native.wgsl's
   native_vertex and native_pixel / native_pixel_blend (and their _u forms for decoded texels) in
   OpenCL C, inlined into native.ll's amdgpu_vs / amdgpu_ps entry points. The same maths, line for
   line where it can be; what differs:
   - one pixel stage for all: the texture's kind (a surface's RGBA8 texels or a decoded texture's)
     and the variant (alpha / 255, or As / 128 for the RGB draw of a two-pass blend) are fields of
     the draw's constants, not separate pipelines;
   - textures are plain memory (texel words in rows of `pitch`), read with loads: native.wgsl reads
     them with textureLoad, never filtered by the sampler (its bilinear mix is its own), and its
     gather (bilinear_u) reads the same four texels as four loads;
   - derivatives from the 2x2 quad with ds_swizzle (GCN 1.1 has no DPP);
   - replacements (texture packs, the sharp font: flag 2048) are not drawn yet: render_ps4.c clears
     the flag and the decoded texture is drawn. */
typedef uint uint32_t;
typedef int int32_t;
typedef ulong uint64_t;

/* dhd_render.h's DhdrNativeConstants (native.wgsl's Native), then render_ps4.c's own fields */
typedef struct {
    float target[2];
    uint32_t flags, atst, aref;
    float alpha_scale, texa_alpha;
    uint32_t scale;
    float fog[4];
    int32_t tex_offset[2];
    uint32_t tex_size[2], wrap[4], region[4];
    float mesh_offset[4], mesh_clip[4], mesh_fog[4];
    uint64_t texture;            /* texel words, or 0 */
    uint32_t tex_w, tex_h, tex_pitch;
    uint32_t tex_kind;           /* 0 none, 1 a surface (texel()), 2 decoded texels (texel_u()) */
    uint32_t variant;            /* 0 alpha / 255, 1 As / 128 (the RGB draw of a two-pass blend) */
    uint32_t pad;
} Nat;

typedef float vec4f __attribute__((ext_vector_type(4)));

static inline float as_f(uint32_t v) { return __builtin_astype(v, float); }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float mixf(float a, float b, float t) { return a * (1.0f - t) + b * t; }
static inline float saturate(float v) { return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v; }

/* ---- the vertex stage (native_vertex) ---- */

__attribute__((always_inline)) void native_vertex_body(uint32_t vertex, uint32_t first, uint32_t indexed,
                                                       global const uint32_t *vertices, global const uint32_t *indices,
                                                       global const Nat *nat, float *out)
{
    uint32_t i = first + vertex;
    uint32_t v = indexed ? indices[i] : i;
    global const uint32_t *w = vertices + v * 8u;
    float x = as_f(w[0]), y = as_f(w[1]), z = as_f(w[2]);
    uint32_t flags = nat->flags, scale = nat->scale;
    /* upscaled sprites sampled by whole texels cover whole GS pixels (finding 196) */
    if ((flags & 1024u) && !(flags & 32u) && (scale & 255u) > 1u) {
        x = __builtin_ceilf(x - 0.5f);
        y = __builtin_ceilf(y - 0.5f);
    }
    out[0] = x / nat->target[0] * 2.0f + -1.0f;
    out[1] = y / nat->target[1] * -2.0f + 1.0f;
    out[2] = z;
    out[3] = 1.0f;
    out[4] = as_f(w[3]); out[5] = as_f(w[4]); out[6] = as_f(w[5]); out[7] = as_f(w[6]);
    uint32_t rgba = w[7];
    out[8] = (float)(rgba & 255u); out[9] = (float)((rgba >> 8) & 255u);
    out[10] = (float)((rgba >> 16) & 255u); out[11] = (float)(rgba >> 24);
}

/* ---- the pixel stage (native_pixel) ---- */

static inline uint32_t wrap_coordinate(int32_t c, uint32_t mode, uint32_t size, uint32_t low, uint32_t high)
{
    if (mode == 0u) return (uint32_t)c & (size - 1u);
    if (mode == 1u) return (uint32_t)(c < 0 ? 0 : c > (int32_t)size - 1 ? (int32_t)size - 1 : c);
    if (mode == 2u) return (uint32_t)(c < (int32_t)low ? (int32_t)low : c > (int32_t)high ? (int32_t)high : c);
    return ((uint32_t)c & low) | high;
}

/* GS texel (cx, cy)'s texel in the texture (texel_xy) */
static inline void texel_xy(global const Nat *nat, int32_t cx, int32_t cy, int32_t *tx, int32_t *ty)
{
    uint32_t S = nat->scale >> 8 & 255u, E = nat->scale >> 16 & 255u;
    if (!E) E = 1;
    if (!S) S = 1;
    uint32_t step = S / E;
    int32_t wx = (int32_t)__builtin_floorf((float)cx / (float)E), wy = (int32_t)__builtin_floorf((float)cy / (float)E);
    int32_t subx = cx - wx * (int32_t)E, suby = cy - wy * (int32_t)E;
    uint32_t ax = wrap_coordinate(wx, nat->wrap[0], nat->tex_size[0], nat->region[0], nat->region[1]);
    uint32_t ay = wrap_coordinate(wy, nat->wrap[1], nat->tex_size[1], nat->region[2], nat->region[3]);
    *tx = (nat->tex_offset[0] + (int32_t)ax) * (int32_t)S + subx * (int32_t)step + (int32_t)(step / 2u);
    *ty = (nat->tex_offset[1] + (int32_t)ay) * (int32_t)S + suby * (int32_t)step + (int32_t)(step / 2u);
}

/* texel() / texel_u(): the texel's colour in GS units (0 outside the texture) */
static inline vec4f texel(global const Nat *nat, int32_t cx, int32_t cy)
{
    int32_t x, y;
    texel_xy(nat, cx, cy, &x, &y);
    uint32_t w = 0;
    if (x >= 0 && y >= 0 && (uint32_t)x < nat->tex_w && (uint32_t)y < nat->tex_h)
        w = ((global const uint32_t *)nat->texture)[(uint32_t)y * nat->tex_pitch + (uint32_t)x];
    vec4f t;
    t.x = (float)(w & 255u); t.y = (float)((w >> 8) & 255u); t.z = (float)((w >> 16) & 255u);
    float a = (float)(w >> 24);
    if (nat->tex_kind == 1u || nat->alpha_scale != 255.0f) a = __builtin_rintf(a / 255.0f * nat->alpha_scale);
    t.w = (nat->flags & 256u) ? nat->texa_alpha : a;
    return t;
}

static inline vec4f mix4(vec4f a, vec4f b, float t)
{
    vec4f r;
    r.x = mixf(a.x, b.x, t); r.y = mixf(a.y, b.y, t); r.z = mixf(a.z, b.z, t); r.w = mixf(a.w, b.w, t);
    return r;
}

static inline vec4f fetch(global const Nat *nat, float u, float v, float across_u, float across_v, float down_u,
                          float down_v, float px, float py)
{
    uint32_t flags = nat->flags;
    uint32_t T = nat->scale & 255u, E = nat->scale >> 16 & 255u;
    if ((flags & 1024u) && !(flags & 32u) && T > 1u && E == 1u) {
        float cx = __builtin_floorf(px / (float)T) * (float)T - px, cy = __builtin_floorf(py / (float)T) * (float)T - py;
        u += across_u * cx + down_u * cy;
        v += across_v * cx + down_v * cy;
    }
    u *= (float)E;
    v *= (float)E;
    if (flags & 32u) {
        u -= 0.5f;
        v -= 0.5f;
        float fu = __builtin_floorf(u), fv = __builtin_floorf(v);
        int32_t bx = (int32_t)fu, by = (int32_t)fv;
        float fx = u - fu, fy = v - fv;
        return mix4(mix4(texel(nat, bx, by), texel(nat, bx + 1, by), fx),
                    mix4(texel(nat, bx, by + 1), texel(nat, bx + 1, by + 1), fx), fy);
    }
    return texel(nat, (int32_t)__builtin_floorf(u), (int32_t)__builtin_floorf(v));
}

/* v of the quad's lane given by the swizzle pattern (quad-permute mode: every lane reads one lane) */
#define quad_lane(v, pattern) __builtin_astype(__builtin_amdgcn_ds_swizzle(__builtin_astype((v), int), (pattern)), float)

/* native_pixel: 1 and the colour in out (0..1, alpha / 255 or / 128) to export; 0 discards. */
__attribute__((always_inline)) int native_pixel_body(global const Nat *nat, float s, float t, float q, float f,
                                                     float r, float g, float b, float a_in, float px, float py, float *out)
{
    /* texture_uv and its derivatives (the quad's lanes 0, 1 and 2: right and below) */
    float qq = __builtin_fabsf(q) > 1e-30f ? q : 1e-30f;
    float u = s / qq * (float)nat->tex_size[0], v = t / qq * (float)nat->tex_size[1];
    float u0 = quad_lane(u, 0x8000), v0 = quad_lane(v, 0x8000);
    float across_u = quad_lane(u, 0x8055) - u0, across_v = quad_lane(v, 0x8055) - v0;
    float down_u = quad_lane(u, 0x80AA) - u0, down_v = quad_lane(v, 0x80AA) - v0;
    uint32_t flags = nat->flags;
    int textured = (flags & 1u) != 0u;
    vec4f tex = {0.0f, 0.0f, 0.0f, 0.0f};
    if (textured && nat->tex_kind) tex = fetch(nat, u, v, across_u, across_v, down_u, down_v, px, py);
    /* combine */
    uint32_t tfx = (flags >> 2) & 3u;
    float a = a_in;
    if (textured && (flags & 2u)) {
        if (tfx == 0u) a = minf(__builtin_floorf(tex.w * a_in / 128.0f), 255.0f);
        else if (tfx == 2u) a = minf(tex.w + a_in, 255.0f);
        else a = tex.w;
    }
    if (flags & 64u) {
        uint32_t ua = (uint32_t)a, ref = nat->aref, test = nat->atst;
        int passed = test == 1u || (test == 2u && ua < ref) || (test == 3u && ua <= ref) || (test == 4u && ua == ref) ||
                     (test == 5u && ua >= ref) || (test == 6u && ua > ref) || (test == 7u && ua != ref);
        if (passed == ((flags & 128u) != 0u)) return 0;
    }
    if (textured) {
        if (tfx == 1u) { r = tex.x; g = tex.y; b = tex.z; }
        else {
            r = minf(__builtin_floorf(tex.x * r / 128.0f), 255.0f);
            g = minf(__builtin_floorf(tex.y * g / 128.0f), 255.0f);
            b = minf(__builtin_floorf(tex.z * b / 128.0f), 255.0f);
        }
        if (tfx >= 2u) { r = minf(r + a_in, 255.0f); g = minf(g + a_in, 255.0f); b = minf(b + a_in, 255.0f); }
    }
    if (flags & 16u) {
        r = __builtin_floorf(mixf(nat->fog[0], r, f));
        g = __builtin_floorf(mixf(nat->fog[1], g, f));
        b = __builtin_floorf(mixf(nat->fog[2], b, f));
    }
    if (flags & 512u) {
        r = __builtin_floorf(r * a / 128.0f);
        g = __builtin_floorf(g * a / 128.0f);
        b = __builtin_floorf(b * a / 128.0f);
    }
    out[0] = r / 255.0f; out[1] = g / 255.0f; out[2] = b / 255.0f;
    out[3] = saturate(nat->variant ? a / 128.0f : a / 255.0f);
    return 1;
}
