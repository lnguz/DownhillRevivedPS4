/* The pixel stage's body for scene.ll (gnm.c's graphics check): written in OpenCL C, compiled to
   LLVM IR and inlined into the amdgpu_ps entry, as the native renderer's shaders will be. A 64x64
   RGBA8 texture, nearest texel at (u, v), with the vertices' alpha. */
typedef float vec4f __attribute__((ext_vector_type(4)));

__attribute__((always_inline)) vec4f scene_shade(float u, float v, float a, global const uint *texture)
{
    int x = (int)(u * 64.0f), y = (int)(v * 64.0f);
    x = x < 0 ? 0 : x > 63 ? 63 : x;
    y = y < 0 ? 0 : y > 63 ? 63 : y;
    uint t = texture[y * 64 + x];
    vec4f c;
    c.x = (float)(t & 255u) / 255.0f;
    c.y = (float)((t >> 8) & 255u) / 255.0f;
    c.z = (float)((t >> 16) & 255u) / 255.0f;
    c.w = a;
    return c;
}
