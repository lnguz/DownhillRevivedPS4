/* The PS4's GPU for compute work (port/ps4/gnm.c): GPU-visible memory, the kernels LLVM built
   (port/ps4/kernels, port/ps4/tools/gcn_kernels.py), and command buffers of dispatches submitted to
   the graphics ring and waited for with a fence. CPU and GPU share addresses: a pointer from
   gnm_alloc is the GPU's address too. */
#ifndef PORT_PS4_GNM_H
#define PORT_PS4_GNM_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t entry;              /* the code's address (256-byte aligned) */
    uint32_t rsrc1, rsrc2;       /* COMPUTE_PGM_RSRC1/2 from the kernel descriptor */
    uint16_t properties;         /* the descriptor's kernel_code_properties: which user SGPRs it takes */
    uint32_t kernarg_size;
} GnmKernel;

/* 1 when the GPU path runs: memory, the kernels' image, and the start-up check passed. */
int gnm_init(void);
/* GPU-visible, CPU-coherent (onion) memory from the pool; never freed. NULL when the pool is spent. */
void *gnm_alloc(size_t size, size_t align);
/* GPU-visible garlic memory: fast for the GPU, write-combined and uncached for the CPU (write it in
   order, never read it). NULL when the pool is spent. */
void *gnm_alloc_garlic(size_t size, size_t align);
/* The kernel by name, 0 when the image has none. */
int gnm_kernel(const char *name, GnmKernel *kernel);
/* A command buffer: dispatches in order, one wait at the end. args (size bytes) is copied into the
   buffer's own argument memory. groups_* are workgroups; threads_x/y each workgroup's size. */
void gnm_begin(void);
int gnm_dispatch(const GnmKernel *k, const void *args, uint32_t size, uint32_t groups_x, uint32_t groups_y,
                 uint32_t threads_x, uint32_t threads_y);
/* The dispatches so far finish before the next one starts, and it sees their writes. */
void gnm_barrier(void);
/* 1 while the buffer has room for another dispatch (and its barrier). */
int gnm_room(void);
/* Submits what gnm_begin started and waits until the GPU has finished it and its writes reached
   memory. 0 on a timeout (the GPU path then turns itself off). */
int gnm_finish(void);
/* ---- graphics (the native renderer) ---- */
typedef struct {
    uint64_t address;            /* the code (256-byte aligned) */
    uint32_t rsrc1, rsrc2;       /* SPI_SHADER_PGM_RSRC1/2 from LLVM (USER_SGPR is the caller's) */
    uint32_t ps_input_ena, ps_input_addr;
} GnmShader;
/* A graphics stage by name (port/ps4/kernels/graphics). 0 when there is none. */
int gnm_graphics_shader(const char *name, GnmShader *shader);
/* Packets into the open buffer: context, graphics SH and UCONFIG registers (offsets from Mesa's
   gfx7.json), the driver's default draw state, an auto-indexed draw of count vertices. */
void gnm_context_reg(uint32_t reg, uint32_t value);
void gnm_context_regs(uint32_t reg, const uint32_t *values, uint32_t n);
void gnm_graphics_sh(uint32_t reg, const uint32_t *values, uint32_t n);
void gnm_uconfig_reg(uint32_t reg, uint32_t value);
void gnm_draw_state_init(void);
void gnm_draw_auto(uint32_t count);
/* All earlier work done and its writes in memory, every cache fresh, before later work starts. */
void gnm_full_barrier(void);
/* GPU memory that can be freed (direct memory, onion or garlic, 64 KB granules); NULL when there is
   none. Any thread. */
void *gnm_memory_alloc(size_t size, int garlic);
void gnm_memory_free(void *address);

/* Counts for the log: the time gnm_finish waited for the GPU (microseconds) and its submits. */
extern uint64_t gnm_wait_us, gnm_submits, gnm_gpu_ticks;  /* gnm_gpu_ticks: the GPU's own clock, start to end */
/* The GPU's clock (100 MHz) once the work before this point is done, for profiles: gnm_mark puts a
   mark in the buffer, gnm_mark_time reads it after gnm_finish (index < 4096). */
void gnm_mark(unsigned index);
uint64_t gnm_mark_time(unsigned index);
/* Once a frame: tells the system the frame's submissions are done (sceGnmSubmitDone). */
void gnm_frame_done(void);
#endif
