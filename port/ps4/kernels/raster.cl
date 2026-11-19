/* The GS renderer's kernels on the PS4's GPU (gpu_cpu.c's GPU device): raster.wgsl's entry points
   over raster_core.h, which the CPU device shares, and a word copy for the device's buffer copies.
   OpenCL C 1.2 (pointers without a qualifier are private), built by port/ps4/tools/gcn_kernels.py. */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

typedef uint uint32_t;
typedef int int32_t;
typedef ulong uint64_t;

/* gpu_exact.inc's Constants (raster.wgsl's State) and Primitive, word for word */
typedef struct { uint32_t reg[128][4], info[4], palette[256]; } Constants;
typedef struct {
    uint32_t position[3][4], uv[3][4];
    float stq[3][4];
    uint32_t info[4], bounds[4];
    uint32_t depth[8];
} Primitive;

#define RC_MEM global
#define RC_INLINE __attribute__((always_inline))
#define RC_LOAD(p) (*(p))
#define RC_STORE(p, v) (*(p) = (v))
#define RC_CAS(p, expected, desired) __sync_val_compare_and_swap((p), (expected), (desired))
static inline int32_t rc_abs(int32_t v) { return v < 0 ? -v : v; }
#include "../raster_core.h"

typedef struct { uint32_t c[12]; } Words12;

/* raster.wgsl's main: one 8x8 workgroup per touched tile of the batch (tile list entries at base). */
kernel void raster(global const uint32_t *bins, global const Primitive *primitives, global const Constants *states,
                   global uint32_t *vram, global const uint32_t *snapshot, uint32_t touched, uint32_t base,
                   uint32_t columns, uint32_t mode)
{
    uint32_t n = __builtin_amdgcn_workgroup_id_y() * columns + __builtin_amdgcn_workgroup_id_x();
    if (n >= touched) return;
    uint32_t entry = base + n * 3u;
    uint32_t tile = bins[entry], first = bins[entry + 1], count = bins[entry + 2];
    uint32_t x0 = (tile & 0xFFFFu) * 8u, y0 = (tile >> 16) * 8u;
    int32_t x = (int32_t)(x0 + __builtin_amdgcn_workitem_id_x()), y = (int32_t)(y0 + __builtin_amdgcn_workitem_id_y());
    uint32_t lane = __builtin_amdgcn_workitem_id_y() * 8u + __builtin_amdgcn_workitem_id_x();
    /* The tile's primitives 64 at a time: each lane loads one and works out its box in the tile
       (into LDS), then every lane goes through the 64 boxes and draws those over its pixel, in
       order. A tile can have hundreds of primitives; one at a time, each waited for its loads. */
    PixelCache cache = {{0u, 0u}, {0u, 0u}};
    local int32_t boxes[64][4];
    local float areas[64];
    local int32_t signs[64];
    local uint32_t ids[64];
    for (uint32_t i0 = 0; i0 < count; i0 += 64u) {
        uint32_t k = i0 + lane;
        if (k < count) {
            uint32_t id = bins[first + k];
            Setup setup;
            int32_t box[4];
            if (!prim_box(&primitives[id], x0, y0, &setup, box)) box[2] = box[0];  /* nothing in the tile */
            boxes[lane][0] = box[0]; boxes[lane][1] = box[1]; boxes[lane][2] = box[2]; boxes[lane][3] = box[3];
            areas[lane] = setup.area;
            signs[lane] = setup.sign;
            ids[lane] = id;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
        uint32_t n = count - i0 < 64u ? count - i0 : 64u;
        /* the lanes together through the 64 boxes: a box no lane is in costs a few LDS reads */
        for (uint32_t j = 0; j < n; j++) {
            if (x < boxes[j][0] || y < boxes[j][1] || x >= boxes[j][2] || y >= boxes[j][3]) continue;
            if (mode == 1u) continue;  /* experiment: everything but the pixel's work */
            Setup setup;
            setup.area = areas[j];
            setup.sign = signs[j];
            global const Primitive *d = &primitives[ids[j]];
            draw_pixel(vram, snapshot, d, &states[d->info[3]], &setup, &cache, (uint32_t)x, (uint32_t)y);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    }
}

/* words of from into to; each of the dispatch's threads takes every stride-th word */
kernel void copy_words(global const uint32_t *from, global uint32_t *to, uint32_t words, uint32_t stride)
{
    for (uint32_t i = __builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x(); i < words; i += stride)
        to[i] = from[i];
}

/* raster.wgsl's local_copy: 8x8 workgroups over the rectangle (k: gpu_exact.inc's copy constants) */
kernel void local_copy(global uint32_t *vram, Words12 k)
{
    uint32_t x = __builtin_amdgcn_workgroup_id_x() * 8u + __builtin_amdgcn_workitem_id_x();
    uint32_t y = __builtin_amdgcn_workgroup_id_y() * 8u + __builtin_amdgcn_workitem_id_y();
    if (x >= k.c[8] || y >= k.c[9]) return;
    uint32_t v = read_pixel(vram, k.c[0], k.c[1], k.c[2], (k.c[3] + x) & 2047u, (k.c[4] + y) & 2047u);
    write_pixel(vram, k.c[5], k.c[6], k.c[7], (k.c[10] + x) & 2047u, (k.c[11] + y) & 2047u, v, 0u);
}

/* raster.wgsl's scanout, one source line per row of out (gpu_cpu.c doubles a field's lines) */
kernel void scanout(global const uint32_t *vram, global uint32_t *out, Words12 k, uint32_t lines)
{
    uint32_t x = __builtin_amdgcn_workgroup_id_x() * 8u + __builtin_amdgcn_workitem_id_x();
    uint32_t y = __builtin_amdgcn_workgroup_id_y() * 8u + __builtin_amdgcn_workitem_id_y();
    if (x >= k.c[5] || y >= lines) return;
    out[y * k.c[5] + x] = scan_pixel(vram, k.c, x, y);
}
