/* The native renderer's compute work on the PS4's GPU (port/ps4/render_ps4.c): surfaces are RGBA8
   texels (one word, red in the low byte) in rows of `pitch` words; what native.wgsl's blit, upload
   and scanout passes and the Rust renderer's texture copies do. 8x8 workgroups, one texel each. */
typedef uint uint32_t;
typedef int int32_t;

#define X (__builtin_amdgcn_workgroup_id_x() * 8u + __builtin_amdgcn_workitem_id_x())
#define Y (__builtin_amdgcn_workgroup_id_y() * 8u + __builtin_amdgcn_workitem_id_y())

/* w x h texels from (sx, sy) of one surface to (dx, dy) of another (or a scratch copy of one). */
kernel void native_copy(global const uint32_t *from, uint32_t from_pitch, global uint32_t *to, uint32_t to_pitch,
                        uint32_t sx, uint32_t sy, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h)
{
    uint32_t x = X, y = Y;
    if (x >= w || y >= h) return;
    to[(dy + y) * to_pitch + dx + x] = from[(sy + y) * from_pitch + sx + x];
}

/* native_blit: target texels [dx, dx + w) x [dy, dy + h) at scale t read the source (scale s, w_s x
   h_s texels) at floor(((p + 0.5) / t + offset) * s), offset the GS pixel distance source - target. */
kernel void native_blit(global const uint32_t *from, uint32_t from_pitch, uint32_t from_w, uint32_t from_h,
                        global uint32_t *to, uint32_t to_pitch, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h,
                        float offset_x, float offset_y, float t, float s)
{
    uint32_t x = X, y = Y;
    if (x >= w || y >= h) return;
    float px = (float)(dx + x) + 0.5f, py = (float)(dy + y) + 0.5f;
    int32_t ux = (int32_t)__builtin_floorf((px / t + offset_x) * s), uy = (int32_t)__builtin_floorf((py / t + offset_y) * s);
    uint32_t v = 0;
    if (ux >= 0 && uy >= 0 && (uint32_t)ux < from_w && (uint32_t)uy < from_h) v = from[(uint32_t)uy * from_pitch + (uint32_t)ux];
    to[(dy + y) * to_pitch + dx + x] = v;
}

/* IMPORT: w x h CPU pixels (GS pixels at (x0, y0)) into a surface at scale s, a block per pixel. */
kernel void native_import(global const uint32_t *pixels, uint32_t w, uint32_t h, uint32_t x0, uint32_t y0, uint32_t s,
                          global uint32_t *to, uint32_t to_pitch, uint32_t to_w, uint32_t to_h)
{
    uint32_t x = X, y = Y;
    if (x >= w * s || y >= h * s) return;
    uint32_t tx = x0 * s + x, ty = y0 * s + y;
    if (tx >= to_w || ty >= to_h) return;
    to[ty * to_pitch + tx] = pixels[(y / s) * w + x / s];
}

/* ALPHA_UPLOAD (native_upload_alpha): a 64 x 32 page of bits (red: alpha bits set, green: alpha
   bits cleared) over the surface's alpha at GS pixel (x0, y0), scale s. */
kernel void native_alpha(global const uint32_t *page, uint32_t x0, uint32_t y0, uint32_t s,
                         global uint32_t *to, uint32_t to_pitch, uint32_t to_w, uint32_t to_h)
{
    uint32_t x = X, y = Y;
    if (x >= 64u * s || y >= 32u * s) return;
    uint32_t tx = x0 * s + x, ty = y0 * s + y;
    if (tx >= to_w || ty >= to_h) return;
    uint32_t bits = page[(y / s) * 64u + x / s];
    uint32_t set = bits & 255u, clear = (bits >> 8) & 255u;
    uint32_t old = to[ty * to_pitch + tx];
    uint32_t alpha = ((old >> 24) & ~clear & 255u) | set;
    to[ty * to_pitch + tx] = (old & 0xFFFFFFu) | (alpha << 24);
}

/* native_scanout: the display picture (width x height texels) from the display surface (scale s):
   texel (tx, ty) shows source ((origin + at / s) & 2047) * s + at % s, at = (tx, ty >> interlace). */
kernel void native_scanout(global const uint32_t *from, uint32_t from_pitch, uint32_t from_w, uint32_t from_h,
                           global uint32_t *to, uint32_t width, uint32_t height, uint32_t origin_x, uint32_t origin_y,
                           uint32_t interlace, uint32_t s)
{
    uint32_t x = X, y = Y;
    if (x >= width || y >= height) return;
    uint32_t ax = x, ay = y >> interlace;
    uint32_t ux = ((origin_x + ax / s) & 2047u) * s + ax % s, uy = ((origin_y + ay / s) & 2047u) * s + ay % s;
    uint32_t v = 0;
    if (ux < from_w && uy < from_h) v = from[uy * from_pitch + ux];
    to[y * width + x] = (v & 0xFFFFFFu) | 0xFF000000u;
}

/* A grown depth surface (32-bit, 1D-tiled: 8x8 micro tiles of 64 words in row-major tile order):
   each old tile into the new surface's tile at the same place, whatever its order inside. */
kernel void native_depth_grow(global const uint32_t *from, uint32_t from_tiles_x, uint32_t from_tiles_y,
                              global uint32_t *to, uint32_t to_tiles_x)
{
    uint32_t tile = __builtin_amdgcn_workgroup_id_x(), word = __builtin_amdgcn_workitem_id_x();
    if (tile >= from_tiles_x * from_tiles_y) return;
    uint32_t tx = tile % from_tiles_x, ty = tile / from_tiles_x;
    to[(ty * to_tiles_x + tx) * 64u + word] = from[tile * 64u + word];
}
