/* The PS4's GPU for compute work: see gnm.h.

   Memory: one pool of direct memory, write-back onion (coherent with the CPU's caches), mapped for
   the CPU and the GPU at the same address.

   Commands (PM4 type-3 packets, as the console's GPU, AMD GCN 1.1, reads them; the packet and
   register layout as AMD's Sea Islands documents and the shadPS4 emulator describe them): a buffer
   starts with sceGnmDispatchInitDefaultHardwareState's packets and an ACQUIRE_MEM that invalidates
   the GPU's caches (the CPU may have written anything since); each dispatch sets COMPUTE_PGM_LO/HI,
   COMPUTE_PGM_RSRC1/2, COMPUTE_NUM_THREAD_X/Y/Z and the user SGPRs the kernel descriptor asks for
   (its kernel arguments' address among them), then DISPATCH_DIRECT; the buffer ends with a CS partial
   flush, an ACQUIRE_MEM that writes the L2 back to memory, and an end-of-pipe event that writes the
   fence value, which the CPU polls. */
#include "gnm.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>

#include <orbis/libkernel.h>

#include "gcn_kernels.h"  /* generated: gcn_image, gcn_kernels */

/* the console's GNM driver (OpenOrbis's header declares these without parameters) */
uint32_t sceGnmDispatchInitDefaultHardwareState(uint32_t *commands, uint32_t dwords);
int32_t sceGnmSubmitCommandBuffers(uint32_t count, const uint32_t *dcb[], uint32_t *dcb_bytes,
                                   const uint32_t *ccb[], uint32_t *ccb_bytes);
int32_t sceGnmSubmitDone(void);

#define POOL_SIZE (96u << 20)
#define COMMAND_WORDS (1u << 20)   /* 4 MB of packets per buffer */
#define ARGUMENT_BYTES (8u << 20)  /* kernel arguments per buffer */

static struct {
    uint8_t *base;
    size_t used;
    int ready, failed;
    uint8_t *image;
    uint32_t *commands, count;
    uint8_t *arguments;
    uint32_t argument_used;
    volatile uint32_t *fence;
    uint32_t fence_value;
    uint8_t *zero;                 /* 64 zero bytes for user SGPRs a kernel takes but never reads */
    volatile uint32_t *marks;      /* garlic (uncached for the CPU): the fence and the start marks */
    uint8_t *garlic;               /* the garlic pool: marks first, then gnm_alloc_garlic's */
    size_t garlic_used;
} gnm;

#define GARLIC_SIZE (64u << 20)

void *gnm_alloc_garlic(size_t size, size_t align)
{
    size_t at = (gnm.garlic_used + align - 1) & ~(align - 1);
    if (!gnm.garlic || at + size > GARLIC_SIZE) return NULL;
    gnm.garlic_used = at + size;
    return gnm.garlic + at;
}

uint64_t gnm_wait_us, gnm_submits, gnm_gpu_ticks;

/* the GPU's clock into marks[8 + 2 * slot] once everything before it in the buffer is done */
static void emit(uint32_t word);
static void timestamp(unsigned slot);

void *gnm_alloc(size_t size, size_t align)
{
    size_t at = (gnm.used + align - 1) & ~(align - 1);
    if (!gnm.base || at + size > POOL_SIZE) return NULL;
    gnm.used = at + size;
    return gnm.base + at;
}

int gnm_kernel(const char *name, GnmKernel *k)
{
    for (size_t i = 0; i < sizeof gcn_kernels / sizeof gcn_kernels[0]; i++) {
        if (strcmp(gcn_kernels[i].name, name)) continue;
        const uint8_t *d = gnm.image + gcn_kernels[i].descriptor;
        int64_t entry_offset;
        memcpy(&k->kernarg_size, d + 8, 4);
        memcpy(&entry_offset, d + 16, 8);
        memcpy(&k->rsrc1, d + 48, 4);
        memcpy(&k->rsrc2, d + 52, 4);
        memcpy(&k->properties, d + 56, 2);
        k->entry = (uint64_t)(uintptr_t)d + (uint64_t)entry_offset;
        uint32_t scratch, lds;
        memcpy(&scratch, d + 4, 4);
        memcpy(&lds, d, 4);
        k->rsrc2 |= ((lds + 511) / 512) << 15;  /* LDS_SIZE, bits 15-23 */
        if (scratch || (k->entry & 255)) {
            fprintf(stderr, "gnm: kernel %s needs scratch memory (%u bytes) or is misplaced\n", name, scratch);
            return 0;
        }
        return 1;
    }
    return 0;
}

/* a type-3 packet header; dwords: the body's words after the header */
#define PKT3(op, dwords, compute) ((3u << 30) | (((dwords) - 1u) << 16) | ((uint32_t)(op) << 8) | ((compute) << 1))
enum { IT_WRITE_DATA = 0x37, IT_NOP = 0x10, IT_DISPATCH_DIRECT = 0x15, IT_EVENT_WRITE = 0x46, IT_EVENT_WRITE_EOP = 0x47,
       IT_ACQUIRE_MEM = 0x58, IT_SET_SH_REG = 0x76 };
enum { COMPUTE_START_X = 0x204, COMPUTE_NUM_THREAD_X = 0x207, COMPUTE_PGM_LO = 0x20C, COMPUTE_PGM_RSRC1 = 0x212, COMPUTE_USER_DATA_0 = 0x240 };

static void emit(uint32_t word) { gnm.commands[gnm.count++] = word; }

static void set_sh(uint32_t reg, const uint32_t *values, uint32_t n)
{
    emit(PKT3(IT_SET_SH_REG, n + 1, 1));
    emit(reg);
    for (uint32_t i = 0; i < n; i++) emit(values[i]);
}

/* ACQUIRE_MEM over all memory: TC (L2) and TCL1 actions, the shader K and I caches. On GCN 1.1 the
   TC action writes the L2 back and invalidates it. */
static void acquire_mem(void)
{
    emit(PKT3(IT_ACQUIRE_MEM, 6, 0));
    emit((1u << 29) | (1u << 27) | (1u << 23) | (1u << 22));
    emit(0xFFFFFFFFu);
    emit(0xFFu);
    emit(0);
    emit(0);
    emit(0x0000000Au);
}

/* One command buffer is built at a time (gnm_begin to gnm_finish): the native renderer's thread and
   the GS worker (the exact renderer's Z probe) both submit. */
static pthread_mutex_t building = PTHREAD_MUTEX_INITIALIZER;

void gnm_begin(void)
{
    pthread_mutex_lock(&building);
    gnm.count = 0;
    gnm.argument_used = 0;
    gnm.count += sceGnmDispatchInitDefaultHardwareState(gnm.commands, 256);
    acquire_mem();
    static const uint32_t start[3] = {0, 0, 0};
    set_sh(COMPUTE_START_X, start, 3);
    timestamp(0);
    /* WRITE_DATA (memory, write confirmed, by the ME): the buffer's number into marks[1], so a
       buffer the GPU never finished still shows whether it began */
    uint64_t mark = (uint64_t)(uintptr_t)&gnm.marks[1];
    emit(PKT3(IT_WRITE_DATA, 4, 0));
    emit((5u << 8) | (1u << 20));
    emit((uint32_t)mark);
    emit((uint32_t)(mark >> 32));
    emit(gnm.fence_value + 1);
}

int gnm_dispatch(const GnmKernel *k, const void *args, uint32_t size, uint32_t groups_x, uint32_t groups_y,
                 uint32_t threads_x, uint32_t threads_y)
{
    if (gnm.count + 64 > COMMAND_WORDS || gnm.argument_used + size + 256 > ARGUMENT_BYTES) return 0;
    uint8_t *arguments = gnm.arguments + gnm.argument_used;
    memcpy(arguments, args, size);
    gnm.argument_used = (gnm.argument_used + size + 255) & ~255u;
    uint32_t program[2] = {(uint32_t)(k->entry >> 8), (uint32_t)(k->entry >> 40)};
    uint32_t resources[2] = {k->rsrc1, k->rsrc2};
    uint32_t threads[3] = {threads_x, threads_y, 1};
    set_sh(COMPUTE_PGM_LO, program, 2);
    set_sh(COMPUTE_PGM_RSRC1, resources, 2);
    set_sh(COMPUTE_NUM_THREAD_X, threads, 3);
    /* the HSA user SGPRs in their order, those the descriptor enables */
    uint32_t user[16], n = 0;
    uint64_t zero = (uint64_t)(uintptr_t)gnm.zero, kernarg = (uint64_t)(uintptr_t)arguments;
    if (k->properties & 1) { user[n++] = 0; user[n++] = 0; user[n++] = 0; user[n++] = 0; }  /* private segment buffer */
    if (k->properties & 2) { user[n++] = (uint32_t)zero; user[n++] = (uint32_t)(zero >> 32); }  /* dispatch packet */
    if (k->properties & 4) { user[n++] = (uint32_t)zero; user[n++] = (uint32_t)(zero >> 32); }  /* queue */
    if (k->properties & 8) { user[n++] = (uint32_t)kernarg; user[n++] = (uint32_t)(kernarg >> 32); }
    if (k->properties & 16) { user[n++] = 0; user[n++] = 0; }  /* dispatch id */
    if (k->properties & 32) { user[n++] = 0; user[n++] = 0; }  /* flat scratch init */
    if (k->properties & 64) user[n++] = 0;                     /* private segment size */
    if (n) set_sh(COMPUTE_USER_DATA_0, user, n);
    emit(PKT3(IT_DISPATCH_DIRECT, 4, 1));
    emit(groups_x);
    emit(groups_y);
    emit(1);
    emit(1);  /* DISPATCH_INITIATOR: COMPUTE_SHADER_EN */
    return 1;
}

/* Between dispatches that depend on each other: the earlier ones finished, and the shader L1 and
   K caches invalidated, so the next reads what they wrote (in the L2, shared by every CU). */
void gnm_barrier(void)
{
    emit(PKT3(IT_EVENT_WRITE, 1, 0));
    emit(7u | (4u << 8));  /* CS_PARTIAL_FLUSH */
    emit(PKT3(IT_ACQUIRE_MEM, 6, 0));
    emit((1u << 27) | (1u << 22));  /* SH_KCACHE, TCL1 */
    emit(0xFFFFFFFFu);
    emit(0xFFu);
    emit(0);
    emit(0);
    emit(0x0000000Au);
}

int gnm_room(void) { return gnm.count + 256 < COMMAND_WORDS && gnm.argument_used + 4096 < ARGUMENT_BYTES; }

int gnm_finish(void)
{
    /* the dispatches done, their writes in memory, then the fence */
    timestamp(1);
    emit(PKT3(IT_EVENT_WRITE, 1, 0));  /* one body word: the event */
    emit(7u | (4u << 8));  /* CS_PARTIAL_FLUSH */
    acquire_mem();
    uint64_t fence = (uint64_t)(uintptr_t)gnm.fence;
    uint32_t value = ++gnm.fence_value;
    emit(PKT3(IT_EVENT_WRITE_EOP, 5, 0));
    emit(0x14u | (5u << 8));  /* CACHE_FLUSH_AND_INV_TS_EVENT: colour and depth caches too */
    emit((uint32_t)fence);
    emit(((uint32_t)(fence >> 32) & 0xFFFFu) | (1u << 29));  /* DATA_SEL: the low 32 bits */
    emit(value);
    emit(0);
    while (gnm.count & 7) emit(0x80000000u);  /* type-2 packets: padding */
    if (gnm.fence_value == 1) {  /* the first buffer, in the log before the GPU sees it */
        for (uint32_t i = 0; i < gnm.count; i += 8) {
            fprintf(stderr, "gnm: %4u:", i);
            for (uint32_t j = i; j < i + 8 && j < gnm.count; j++) fprintf(stderr, " %08x", gnm.commands[j]);
            fprintf(stderr, "\n");
        }
    }
    __builtin_ia32_sfence();  /* the CPU's write-combined (garlic) writes reach memory before the GPU reads */
    const uint32_t *dcb[1] = {gnm.commands};
    uint32_t bytes[1] = {gnm.count * 4};
    int32_t rc = sceGnmSubmitCommandBuffers(1, dcb, bytes, NULL, NULL);
    if (gnm.fence_value == 1) fprintf(stderr, "gnm: step 4, submitted (%d), waiting for the fence\n", rc);
    if (rc != 0) {
        fprintf(stderr, "gnm: sceGnmSubmitCommandBuffers failed (0x%08x)\n", (unsigned)rc);
        pthread_mutex_unlock(&building);
        return 0;
    }
    uint64_t start = sceKernelGetProcessTime();
    while (*gnm.fence != value) {
        if (sceKernelGetProcessTime() - start > 3000000) {
            fprintf(stderr, "gnm: the GPU did not finish within 3 s (fence %u, want %u; start mark %u)\n",
                    *gnm.fence, value, gnm.marks[1]);
            gnm.failed = 1;
            pthread_mutex_unlock(&building);
            return 0;
        }
        /* spin: the GPU's work takes micro- to milliseconds, and a sleep on the console lasts far
           longer than asked (a field's dozens of waits slept away most of it) */
        __builtin_ia32_pause();
    }
    gnm_wait_us += sceKernelGetProcessTime() - start;
    gnm_submits++;
    uint64_t t0 = gnm.marks[8] | (uint64_t)gnm.marks[9] << 32, t1 = gnm.marks[10] | (uint64_t)gnm.marks[11] << 32;
    if (t1 > t0) gnm_gpu_ticks += t1 - t0;
    pthread_mutex_unlock(&building);
    return 1;
}

static void timestamp(unsigned slot)
{
    uint64_t at = (uint64_t)(uintptr_t)&gnm.marks[8 + 2 * slot];
    emit(PKT3(IT_EVENT_WRITE_EOP, 5, 0));
    emit(0x28u | (5u << 8));  /* BOTTOM_OF_PIPE_TS */
    emit((uint32_t)at);
    emit(((uint32_t)(at >> 32) & 0xFFFFu) | (3u << 29));  /* DATA_SEL 3: the GPU's 64-bit clock */
    emit(0);
    emit(0);
}

/* Profile marks: the GPU's clock after the work so far, into marks[64 + 2 * index] (up to 4096). */
void gnm_mark(unsigned index)
{
    if (index < 4096) timestamp(28 + index);
}

uint64_t gnm_mark_time(unsigned index)
{
    volatile uint32_t *m = &gnm.marks[8 + 2 * (28 + index)];
    return m[0] | (uint64_t)m[1] << 32;
}

void gnm_frame_done(void)
{
    if (gnm.ready) sceGnmSubmitDone();
}

/* The start-up check: the probe kernel over 256 words, against the CPU's answer. */
static int check(void)
{
    GnmKernel probe;
    if (!gnm_kernel("probe", &probe)) { fprintf(stderr, "gnm: no probe kernel\n"); return 0; }
    uint32_t *data = gnm_alloc(256 * 4, 256);
    for (uint32_t i = 0; i < 256; i++) data[i] = i;
    struct { uint64_t data; uint32_t count, add; } args = {(uint64_t)(uintptr_t)data, 256, 7};
    fprintf(stderr, "gnm: probe kernel at %p, rsrc1 0x%08x rsrc2 0x%08x properties 0x%x\n", (void *)(uintptr_t)probe.entry,
            probe.rsrc1, probe.rsrc2, probe.properties);
    uint64_t start = sceKernelGetProcessTime();
    fprintf(stderr, "gnm: step 1, commands at %p\n", (void *)gnm.commands);
    gnm_begin();
    fprintf(stderr, "gnm: step 2, %u words of start-up packets\n", gnm.count);
    gnm_dispatch(&probe, &args, sizeof args, 4, 1, 64, 1);
    fprintf(stderr, "gnm: step 3, %u words, submitting\n", gnm.count);
    int finished = gnm_finish();
    unsigned wrong = 0;
    for (uint32_t i = 0; i < 256; i++) wrong += data[i] != i * 3 + 7;
    fprintf(stderr, "gnm: GPU check %s (%u of 256 words wrong, words 0-3: %u %u %u %u, %llu us)\n",
            finished && !wrong ? "passed" : "FAILED", wrong, data[0], data[1], data[2], data[3],
            (unsigned long long)(sceKernelGetProcessTime() - start));
    return finished && !wrong;
}

/* ---- the graphics pipeline's check (port/ps4/kernels/graphics/triangle.ll) ----

   One red triangle into a 256x256 RGBA8 render target, counted by the CPU: the vertex and pixel
   stages, the colour buffer and the draw, as the native renderer will need them. Context state
   from the driver's default (sceGnmDrawInitDefaultHardwareState350), then the registers below
   (offsets from Mesa's GCN 1.1 register database). Only with /data/dhd/gpu_triangle: a wrong
   graphics state can hang the GPU. */
uint32_t sceGnmDrawInitDefaultHardwareState350(uint32_t *commands, uint32_t dwords);

enum { IT_SET_CONTEXT_REG = 0x69, IT_SET_UCONFIG_REG = 0x79, IT_DRAW_INDEX_AUTO = 0x2D };

static void set_context(uint32_t reg, uint32_t value)
{
    emit(PKT3(IT_SET_CONTEXT_REG, 2, 0));
    emit(reg);
    emit(value);
}

static void set_graphics_sh(uint32_t reg, uint32_t a, uint32_t b)
{
    emit(PKT3(IT_SET_SH_REG, 3, 0));
    emit(reg);
    emit(a);
    emit(b);
}

static int find_shader(const char *name, uint32_t *offset, uint32_t *rsrc1, uint32_t *rsrc2, uint32_t *ena, uint32_t *addr)
{
    for (size_t i = 0; i < sizeof gcn_shaders / sizeof gcn_shaders[0]; i++)
        if (!strcmp(gcn_shaders[i].name, name)) {
            *offset = gcn_shaders[i].offset; *rsrc1 = gcn_shaders[i].rsrc1; *rsrc2 = gcn_shaders[i].rsrc2;
            *ena = gcn_shaders[i].ps_input_ena; *addr = gcn_shaders[i].ps_input_addr;
            return 1;
        }
    return 0;
}

static void triangle_check(void)
{
    struct stat st;
    if (stat("/data/dhd/gpu_triangle", &st) != 0) return;
    uint32_t vs, vs1, vs2, ps, ps1, ps2, ena, addr, unused;
    if (!find_shader("triangle_vs", &vs, &vs1, &vs2, &unused, &unused) ||
        !find_shader("triangle_ps", &ps, &ps1, &ps2, &ena, &addr)) {
        fprintf(stderr, "gnm: triangle: no shaders\n");
        return;
    }
    uint8_t *code = gnm_alloc(sizeof gcn_graphics, 4096);
    uint32_t *target = gnm_alloc(256 * 256 * 4, 65536);
    if (!code || !target) return;
    memcpy(code, gcn_graphics, sizeof gcn_graphics);
    memset(target, 0, 256 * 256 * 4);
    uint64_t vs_at = (uint64_t)(uintptr_t)(code + vs), ps_at = (uint64_t)(uintptr_t)(code + ps);
    uint64_t rt = (uint64_t)(uintptr_t)target;
    fprintf(stderr, "gnm: triangle: vs at %p (rsrc1 0x%x), ps at %p (rsrc1 0x%x, inputs 0x%x/0x%x), target %p\n",
            (void *)(uintptr_t)vs_at, vs1, (void *)(uintptr_t)ps_at, ps1, ena, addr, (void *)target);
    gnm_begin();
    gnm.count += sceGnmDrawInitDefaultHardwareState350(gnm.commands + gnm.count, 256);
    /* the colour buffer: RGBA8 UNORM, linear (tile mode index 8, the display's LinearAligned) */
    set_context(0x318, (uint32_t)(rt >> 8));     /* CB_COLOR0_BASE */
    set_context(0x319, 256 / 8 - 1);             /* CB_COLOR0_PITCH.TILE_MAX */
    set_context(0x31A, 256 * 256 / 64 - 1);      /* CB_COLOR0_SLICE.TILE_MAX */
    set_context(0x31B, 0);                       /* CB_COLOR0_VIEW */
    set_context(0x31C, 10u << 2);                /* CB_COLOR0_INFO: FORMAT COLOR_8_8_8_8, UNORM, SWAP_STD */
    set_context(0x31D, 8);                       /* CB_COLOR0_ATTRIB.TILE_MODE_INDEX */
    set_context(0x202, 0x00CC0010u);             /* CB_COLOR_CONTROL: CB_NORMAL, ROP3 copy */
    set_context(0x1E0, 0);                       /* CB_BLEND0_CONTROL: no blending */
    set_context(0x08E, 0xF);                     /* CB_TARGET_MASK */
    set_context(0x08F, 0xF);                     /* CB_SHADER_MASK */
    /* no depth or stencil */
    set_context(0x010, 0);                       /* DB_Z_INFO: FORMAT INVALID */
    set_context(0x011, 0);                       /* DB_STENCIL_INFO */
    set_context(0x200, 0);                       /* DB_DEPTH_CONTROL */
    set_context(0x000, 0);                       /* DB_RENDER_CONTROL */
    set_context(0x203, 1u << 4);                 /* DB_SHADER_CONTROL: EARLY_Z_THEN_LATE_Z */
    /* the screen: 256x256, screen-space positions (no viewport transform), no clipping or culling */
    set_context(0x00C, 0);                       /* PA_SC_SCREEN_SCISSOR_TL */
    set_context(0x00D, 256u | 256u << 16);       /* PA_SC_SCREEN_SCISSOR_BR */
    set_context(0x080, 0);                       /* PA_SC_WINDOW_OFFSET */
    set_context(0x081, 0x80000000u);             /* PA_SC_WINDOW_SCISSOR_TL: WINDOW_OFFSET_DISABLE */
    set_context(0x082, 256u | 256u << 16);       /* PA_SC_WINDOW_SCISSOR_BR */
    set_context(0x090, 0x80000000u);             /* PA_SC_GENERIC_SCISSOR_TL */
    set_context(0x091, 256u | 256u << 16);       /* PA_SC_GENERIC_SCISSOR_BR */
    set_context(0x094, 0x80000000u);             /* PA_SC_VPORT_SCISSOR_0_TL */
    set_context(0x095, 256u | 256u << 16);       /* PA_SC_VPORT_SCISSOR_0_BR */
    set_context(0x0B4, 0x00000000u);             /* PA_SC_VPORT_ZMIN_0: 0.0 */
    set_context(0x0B5, 0x3F800000u);             /* PA_SC_VPORT_ZMAX_0: 1.0 */
    set_context(0x206, (1u << 8) | (1u << 9) | (1u << 10));  /* PA_CL_VTE_CNTL: XY, Z, W0 as given */
    set_context(0x204, (1u << 16) | (1u << 19)); /* PA_CL_CLIP_CNTL: CLIP_DISABLE, DX_CLIP_SPACE_DEF */
    set_context(0x205, 0);                       /* PA_SU_SC_MODE_CNTL: no culling */
    set_context(0x207, 0);                       /* PA_CL_VS_OUT_CNTL */
    set_context(0x292, 0);                       /* PA_SC_MODE_CNTL_0 */
    set_context(0x2F8, 0);                       /* PA_SC_AA_CONFIG: one sample */
    /* the stages' interface: one parameter, four-component position, a 32-bit colour export */
    set_context(0x1B1, 0);                       /* SPI_VS_OUT_CONFIG: one parameter */
    set_context(0x1C3, 4);                       /* SPI_SHADER_POS_FORMAT: POS0 4COMP */
    set_context(0x191, 0);                       /* SPI_PS_INPUT_CNTL_0 */
    set_context(0x1B3, ena);                     /* SPI_PS_INPUT_ENA */
    set_context(0x1B4, addr);                    /* SPI_PS_INPUT_ADDR */
    set_context(0x1B6, 1);                       /* SPI_PS_IN_CONTROL: NUM_INTERP 1 */
    set_context(0x1B8, 0);                       /* SPI_BARYC_CNTL */
    set_context(0x1C4, 0);                       /* SPI_SHADER_Z_FORMAT */
    set_context(0x1C5, 9);                       /* SPI_SHADER_COL_FORMAT: COL0 32_ABGR */
    set_context(0x2D5, 0);                       /* VGT_SHADER_STAGES_EN: VS and PS only */
    set_context(0x290, 0);                       /* VGT_GS_MODE */
    set_graphics_sh(0x48, (uint32_t)(vs_at >> 8), (uint32_t)(vs_at >> 40));  /* SPI_SHADER_PGM_LO/HI_VS */
    set_graphics_sh(0x4A, vs1, vs2);                                          /* SPI_SHADER_PGM_RSRC1/2_VS */
    set_graphics_sh(0x08, (uint32_t)(ps_at >> 8), (uint32_t)(ps_at >> 40));  /* SPI_SHADER_PGM_LO/HI_PS */
    set_graphics_sh(0x0A, ps1, ps2);                                          /* SPI_SHADER_PGM_RSRC1/2_PS */
    emit(PKT3(IT_SET_UCONFIG_REG, 2, 0));
    emit(0x242);                                 /* VGT_PRIMITIVE_TYPE */
    emit(4);                                     /* DI_PT_TRILIST */
    emit(PKT3(IT_DRAW_INDEX_AUTO, 2, 0));
    emit(3);                                     /* three vertices */
    emit(2);                                     /* DRAW_INITIATOR: SOURCE_SELECT auto index */
    emit(PKT3(IT_EVENT_WRITE, 1, 0));
    emit(0x16u);                                 /* CACHE_FLUSH_AND_INV_EVENT: the colour buffer to memory */
    uint64_t start = sceKernelGetProcessTime();
    int ok = gnm_finish();
    unsigned red = 0, other = 0;
    for (unsigned i = 0; i < 256 * 256; i++) {
        if (target[i] == 0xFF0000FFu) red++;
        else if (target[i]) other++;
    }
    fprintf(stderr, "gnm: triangle check %s: %u red pixels (16200 expected), %u others, pixel (30,30) 0x%08x, %llu us\n",
            ok && red > 15000 && red < 17500 ? "passed" : "FAILED", red, other, target[30 * 256 + 30],
            (unsigned long long)(sceKernelGetProcessTime() - start));
}

/* ---- the graphics pipeline's second check (kernels/graphics/scene.ll, scene.cl) ----

   Three triangles into a 256x256 RGBA8 target with a 32-bit float depth buffer (1D-tiled, as the
   DB requires a tiled surface): A textured (red and green squares) at z 0.5, B blue at z 0.7 behind
   it (A must stay where they overlap), C white at half alpha in front at z 0.2, blended over A. The
   CPU checks four pixels; the log also shows the depth buffer's first words, for its layout. Only
   with /data/dhd/gpu_scene. */
static void scene_check(void)
{
    struct stat st;
    if (stat("/data/dhd/gpu_scene", &st) != 0) return;
    uint32_t vs, vs1, vs2, ps, ps1, ps2, ena, addr, unused;
    if (!find_shader("scene_vs", &vs, &vs1, &vs2, &unused, &unused) ||
        !find_shader("scene_ps", &ps, &ps1, &ps2, &ena, &addr)) {
        fprintf(stderr, "gnm: scene: no shaders\n");
        return;
    }
    uint8_t *code = gnm_alloc(sizeof gcn_graphics, 4096);
    uint32_t *target = gnm_alloc(256 * 256 * 4, 65536);
    float *depth = gnm_alloc(256 * 256 * 4, 65536);
    uint32_t *checker = gnm_alloc(64 * 64 * 4, 256), *blue = gnm_alloc(64 * 64 * 4, 256), *white = gnm_alloc(64 * 64 * 4, 256);
    float *vertices = gnm_alloc(9 * 8 * 4, 256);
    if (!code || !target || !depth || !checker || !blue || !white || !vertices) return;
    memcpy(code, gcn_graphics, sizeof gcn_graphics);
    memset(target, 0, 256 * 256 * 4);
    for (unsigned i = 0; i < 256 * 256; i++) depth[i] = 1.0f;  /* the clear value: any tiling */
    for (unsigned y = 0; y < 64; y++)
        for (unsigned x = 0; x < 64; x++) {
            checker[y * 64 + x] = ((x >> 3) + (y >> 3)) & 1 ? 0xFF00FF00u : 0xFF0000FFu;  /* green : red */
            blue[y * 64 + x] = 0xFFFF0000u;
            white[y * 64 + x] = 0xFFFFFFFFu;
        }
    /* x, y, z, 1, u, v, alpha, 1 */
    static const float tris[3][3][8] = {
        {{16, 16, 0.5f, 1, 0, 0, 1, 1}, {200, 16, 0.5f, 1, 1, 0, 1, 1}, {16, 200, 0.5f, 1, 0, 1, 1, 1}},
        {{100, 100, 0.7f, 1, 0, 0, 1, 1}, {240, 100, 0.7f, 1, 1, 0, 1, 1}, {100, 240, 0.7f, 1, 0, 1, 1, 1}},
        {{40, 40, 0.2f, 1, 0, 0, 0.5f, 1}, {120, 40, 0.2f, 1, 1, 0, 0.5f, 1}, {40, 120, 0.2f, 1, 0, 1, 0.5f, 1}},
    };
    memcpy(vertices, tris, sizeof tris);
    uint64_t vs_at = (uint64_t)(uintptr_t)(code + vs), ps_at = (uint64_t)(uintptr_t)(code + ps);
    uint64_t rt = (uint64_t)(uintptr_t)target, db = (uint64_t)(uintptr_t)depth;
    gnm_begin();
    gnm.count += sceGnmDrawInitDefaultHardwareState350(gnm.commands + gnm.count, 256);
    set_context(0x318, (uint32_t)(rt >> 8));     /* CB_COLOR0_BASE */
    set_context(0x319, 256 / 8 - 1);             /* CB_COLOR0_PITCH */
    set_context(0x31A, 256 * 256 / 64 - 1);      /* CB_COLOR0_SLICE */
    set_context(0x31B, 0);                       /* CB_COLOR0_VIEW */
    set_context(0x31C, 10u << 2);                /* CB_COLOR0_INFO: COLOR_8_8_8_8 UNORM */
    set_context(0x31D, 8);                       /* CB_COLOR0_ATTRIB: LinearAligned */
    set_context(0x202, 0x00CC0010u);             /* CB_COLOR_CONTROL */
    set_context(0x08E, 0xF);                     /* CB_TARGET_MASK */
    set_context(0x08F, 0xF);                     /* CB_SHADER_MASK */
    /* the depth buffer: Z_32_FLOAT, ARRAY_1D_TILED_THIN1, no stencil, no HTILE */
    set_context(0x00F, (2u << 4) | (12u << 8));  /* DB_DEPTH_INFO: 1D thin, P8_32x32_16x16 */
    set_context(0x010, 3u);                      /* DB_Z_INFO: Z_32_FLOAT */
    set_context(0x011, 1u << 29);                /* DB_STENCIL_INFO: invalid, TILE_STENCIL_DISABLE */
    set_context(0x012, (uint32_t)(db >> 8));     /* DB_Z_READ_BASE */
    set_context(0x013, (uint32_t)(db >> 8));     /* DB_STENCIL_READ_BASE */
    set_context(0x014, (uint32_t)(db >> 8));     /* DB_Z_WRITE_BASE */
    set_context(0x015, (uint32_t)(db >> 8));     /* DB_STENCIL_WRITE_BASE */
    set_context(0x016, (256u / 8 - 1) | ((256u / 8 - 1) << 11));  /* DB_DEPTH_SIZE */
    set_context(0x017, 256 * 256 / 64 - 1);      /* DB_DEPTH_SLICE */
    set_context(0x002, 0);                       /* DB_DEPTH_VIEW */
    set_context(0x2AF, 0);                       /* DB_HTILE_SURFACE */
    set_context(0x000, 0);                       /* DB_RENDER_CONTROL */
    set_context(0x003, 0);                       /* DB_RENDER_OVERRIDE */
    set_context(0x203, 1u << 4);                 /* DB_SHADER_CONTROL: EARLY_Z_THEN_LATE_Z */
    set_context(0x00C, 0);                       /* PA_SC_SCREEN_SCISSOR_TL */
    set_context(0x00D, 256u | 256u << 16);       /* PA_SC_SCREEN_SCISSOR_BR */
    set_context(0x080, 0);                       /* PA_SC_WINDOW_OFFSET */
    set_context(0x081, 0x80000000u);             /* PA_SC_WINDOW_SCISSOR_TL */
    set_context(0x082, 256u | 256u << 16);       /* PA_SC_WINDOW_SCISSOR_BR */
    set_context(0x090, 0x80000000u);             /* PA_SC_GENERIC_SCISSOR_TL */
    set_context(0x091, 256u | 256u << 16);       /* PA_SC_GENERIC_SCISSOR_BR */
    set_context(0x094, 0x80000000u);             /* PA_SC_VPORT_SCISSOR_0_TL */
    set_context(0x095, 256u | 256u << 16);       /* PA_SC_VPORT_SCISSOR_0_BR */
    set_context(0x0B4, 0x00000000u);             /* PA_SC_VPORT_ZMIN_0 */
    set_context(0x0B5, 0x3F800000u);             /* PA_SC_VPORT_ZMAX_0 */
    set_context(0x206, (1u << 8) | (1u << 9) | (1u << 10));  /* PA_CL_VTE_CNTL */
    set_context(0x204, (1u << 16) | (1u << 19)); /* PA_CL_CLIP_CNTL */
    set_context(0x205, 0);                       /* PA_SU_SC_MODE_CNTL */
    set_context(0x207, 0);                       /* PA_CL_VS_OUT_CNTL */
    set_context(0x292, 0);                       /* PA_SC_MODE_CNTL_0 */
    set_context(0x2F8, 0);                       /* PA_SC_AA_CONFIG */
    set_context(0x1B1, 0);                       /* SPI_VS_OUT_CONFIG: one parameter */
    set_context(0x1C3, 4);                       /* SPI_SHADER_POS_FORMAT */
    set_context(0x191, 0);                       /* SPI_PS_INPUT_CNTL_0 */
    set_context(0x1B3, ena);                     /* SPI_PS_INPUT_ENA */
    set_context(0x1B4, addr);                    /* SPI_PS_INPUT_ADDR */
    set_context(0x1B6, 1);                       /* SPI_PS_IN_CONTROL: NUM_INTERP 1 */
    set_context(0x1B8, 0);                       /* SPI_BARYC_CNTL */
    set_context(0x1C4, 0);                       /* SPI_SHADER_Z_FORMAT */
    set_context(0x1C5, 4);                       /* SPI_SHADER_COL_FORMAT: FP16_ABGR (blendable) */
    set_context(0x2D5, 0);                       /* VGT_SHADER_STAGES_EN */
    set_context(0x290, 0);                       /* VGT_GS_MODE */
    set_graphics_sh(0x48, (uint32_t)(vs_at >> 8), (uint32_t)(vs_at >> 40));
    /* LLVM's amdgpu_vs/amdgpu_ps (mesa3d) take their inreg arguments from s2 on: s0-s1 are reserved,
       so the user data is four SGPRs, the address in the last two (the pixel stage's PRIM_MASK then
       lands in s4, where it reads it). With two, the stage read its address from garbage. */
    set_graphics_sh(0x4A, vs1, vs2 | (4u << 1));  /* RSRC2_VS.USER_SGPR: 4 */
    set_graphics_sh(0x08, (uint32_t)(ps_at >> 8), (uint32_t)(ps_at >> 40));
    set_graphics_sh(0x0A, ps1, ps2 | (4u << 1));  /* RSRC2_PS.USER_SGPR: 4 */
    emit(PKT3(IT_SET_UCONFIG_REG, 2, 0));
    emit(0x242);
    emit(4);                                     /* VGT_PRIMITIVE_TYPE: DI_PT_TRILIST */
    const uint32_t *textures[3] = {checker, blue, white};
    for (unsigned t = 0; t < 3; t++) {
        uint64_t vb = (uint64_t)(uintptr_t)(vertices + t * 24), tex = (uint64_t)(uintptr_t)textures[t];
        /* A and B: test and write; C: test, no write, blended with its alpha */
        set_context(0x200, t < 2 ? (1u << 1) | (1u << 2) | (1u << 4) : (1u << 1) | (1u << 4));  /* DB_DEPTH_CONTROL: LESS */
        set_context(0x1E0, t < 2 ? 0u : 4u | (5u << 8) | (4u << 16) | (5u << 24) | (1u << 30));  /* CB_BLEND0: SRC_ALPHA, ONE_MINUS */
        set_graphics_sh(0x4C, 0, 0);                                  /* SPI_SHADER_USER_DATA_VS_0/1: reserved */
        set_graphics_sh(0x4E, (uint32_t)vb, (uint32_t)(vb >> 32));    /* SPI_SHADER_USER_DATA_VS_2/3 */
        set_graphics_sh(0x0C, 0, 0);                                  /* SPI_SHADER_USER_DATA_PS_0/1: reserved */
        set_graphics_sh(0x0E, (uint32_t)tex, (uint32_t)(tex >> 32));  /* SPI_SHADER_USER_DATA_PS_2/3 */
        emit(PKT3(IT_DRAW_INDEX_AUTO, 2, 0));
        emit(3);
        emit(2);
    }
    emit(PKT3(IT_EVENT_WRITE, 1, 0));
    emit(0x16u);                                 /* CACHE_FLUSH_AND_INV_EVENT */
    uint64_t start = sceKernelGetProcessTime();
    int ok = gnm_finish();
    uint32_t p1 = target[30 * 256 + 30], p2 = target[105 * 256 + 105], p3 = target[130 * 256 + 130], p4 = target[50 * 256 + 50];
    int good1 = p1 == 0xFF0000FFu, good2 = p2 == 0xFF0000FFu, good3 = p3 == 0xFFFF0000u;
    unsigned r4 = p4 & 255, g4 = (p4 >> 8) & 255, b4 = (p4 >> 16) & 255;
    int good4 = r4 >= 250 && g4 >= 120 && g4 <= 136 && b4 >= 120 && b4 <= 136;
    fprintf(stderr, "gnm: scene check %s: texture 0x%08x %s, depth test 0x%08x %s, behind 0x%08x %s, blend 0x%08x %s (%llu us)\n",
            ok && good1 && good2 && good3 && good4 ? "passed" : "FAILED", p1, good1 ? "ok" : "WRONG", p2, good2 ? "ok" : "WRONG",
            p3, good3 ? "ok" : "WRONG", p4, good4 ? "ok" : "WRONG", (unsigned long long)(sceKernelGetProcessTime() - start));
    unsigned written = 0;
    for (unsigned i = 0; i < 256 * 256; i++) written += depth[i] != 1.0f;
    fprintf(stderr, "gnm: scene depth: %u words written; words 0-15:", written);
    for (unsigned i = 0; i < 16; i++) fprintf(stderr, " %.3f", depth[i]);
    fprintf(stderr, "; row 2 (words 512-527):");
    for (unsigned i = 512; i < 528; i++) fprintf(stderr, " %.3f", depth[i]);
    fprintf(stderr, "\n");
    unsigned colored = 0;
    for (unsigned i = 0; i < 256 * 256; i++) colored += target[i] != 0;
    fprintf(stderr, "gnm: scene: %u pixels drawn\n", colored);
}

/* ---- for the native renderer (port/ps4/render_ps4.c) ---- */

static uint8_t *graphics_code;    /* gcn_graphics, in GPU memory once (gnm_init) */

int gnm_graphics_shader(const char *name, GnmShader *shader)
{
    uint32_t offset;
    if (!graphics_code || !find_shader(name, &offset, &shader->rsrc1, &shader->rsrc2, &shader->ps_input_ena,
                                       &shader->ps_input_addr))
        return 0;
    shader->address = (uint64_t)(uintptr_t)(graphics_code + offset);
    return 1;
}

void gnm_context_reg(uint32_t reg, uint32_t value) { set_context(reg, value); }

void gnm_context_regs(uint32_t reg, const uint32_t *values, uint32_t n)
{
    emit(PKT3(IT_SET_CONTEXT_REG, n + 1, 0));
    emit(reg);
    for (uint32_t i = 0; i < n; i++) emit(values[i]);
}

void gnm_graphics_sh(uint32_t reg, const uint32_t *values, uint32_t n)
{
    emit(PKT3(IT_SET_SH_REG, n + 1, 0));
    emit(reg);
    for (uint32_t i = 0; i < n; i++) emit(values[i]);
}

void gnm_uconfig_reg(uint32_t reg, uint32_t value)
{
    emit(PKT3(IT_SET_UCONFIG_REG, 2, 0));
    emit(reg);
    emit(value);
}

void gnm_draw_state_init(void) { gnm.count += sceGnmDrawInitDefaultHardwareState350(gnm.commands + gnm.count, 256); }

void gnm_draw_auto(uint32_t count)
{
    emit(PKT3(IT_DRAW_INDEX_AUTO, 2, 0));
    emit(count);
    emit(2);  /* DRAW_INITIATOR: SOURCE_SELECT auto index */
}

/* Everything before done and in memory (colour and depth caches flushed, the L2 written back and
   every cache invalidated) before anything after starts: an end-of-pipe event writes a value,
   WAIT_REG_MEM holds the command processor until it is there. Between draws and the compute work
   or draws that read what they drew. */
void gnm_full_barrier(void)
{
    static uint32_t value;
    uint64_t at = (uint64_t)(uintptr_t)&gnm.marks[4];
    value++;
    emit(PKT3(IT_EVENT_WRITE_EOP, 5, 0));
    emit(0x14u | (5u << 8));  /* CACHE_FLUSH_AND_INV_TS_EVENT */
    emit((uint32_t)at);
    emit(((uint32_t)(at >> 32) & 0xFFFFu) | (1u << 29));  /* DATA_SEL: 32 bits */
    emit(value);
    emit(0);
    emit(PKT3(0x3C, 6, 0));  /* WAIT_REG_MEM: memory, equal, by the ME */
    emit(3u | (1u << 4));
    emit((uint32_t)at);
    emit((uint32_t)(at >> 32));
    emit(value);
    emit(0xFFFFFFFFu);
    emit(4);  /* poll interval */
    acquire_mem();
}

/* GPU memory that can be given back: one direct-memory mapping per allocation (64 KB granules). */
typedef struct Mapping { void *address; off_t offset; size_t size; struct Mapping *next; } Mapping;
static Mapping *mappings;
static pthread_mutex_t mapping_lock = PTHREAD_MUTEX_INITIALIZER;

void *gnm_memory_alloc(size_t size, int garlic)
{
    size = (size + 65535) & ~(size_t)65535;
    off_t offset;
    void *address = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, 65536, garlic ? 3 : 0, &offset) != 0)
        return NULL;
    if (sceKernelMapDirectMemory(&address, size, 0x33, 0, offset, 65536) != 0) {
        sceKernelReleaseDirectMemory(offset, size);
        return NULL;
    }
    Mapping *m = malloc(sizeof *m);
    if (!m) { sceKernelReleaseDirectMemory(offset, size); return NULL; }
    m->address = address; m->offset = offset; m->size = size;
    pthread_mutex_lock(&mapping_lock);
    m->next = mappings;
    mappings = m;
    pthread_mutex_unlock(&mapping_lock);
    return address;
}

void gnm_memory_free(void *address)
{
    if (!address) return;
    pthread_mutex_lock(&mapping_lock);
    for (Mapping **link = &mappings; *link; link = &(*link)->next)
        if ((*link)->address == address) {
            Mapping *m = *link;
            *link = m->next;
            pthread_mutex_unlock(&mapping_lock);
            sceKernelReleaseDirectMemory(m->offset, m->size);  /* unmaps it too */
            free(m);
            return;
        }
    pthread_mutex_unlock(&mapping_lock);
}

int gnm_init(void)
{
    if (gnm.ready || gnm.failed) return gnm.ready;
    gnm.failed = 1;
    struct stat st;
    if (stat("/data/dhd/nogpu", &st) == 0) {
        fprintf(stderr, "gnm: off (/data/dhd/nogpu)\n");
        return 0;
    }
    off_t offset;
    void *memory = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), POOL_SIZE, 2u << 20, 0, &offset) != 0 ||
        sceKernelMapDirectMemory(&memory, POOL_SIZE, 0x33, 0, offset, 2u << 20) != 0) {
        fprintf(stderr, "gnm: no GPU memory\n");
        return 0;
    }
    off_t marks_offset;
    void *marks = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), GARLIC_SIZE, 2u << 20, 3, &marks_offset) != 0 ||
        sceKernelMapDirectMemory(&marks, GARLIC_SIZE, 0x33, 0, marks_offset, 2u << 20) != 0) {
        fprintf(stderr, "gnm: no garlic memory\n");
        return 0;
    }
    gnm.marks = marks;
    gnm.garlic = marks;
    gnm.garlic_used = 65536;  /* the marks: fence, start mark, timestamps (profile: 4096 of them) */
    gnm.marks[0] = gnm.marks[1] = 0;
    extern int32_t sceGnmAreSubmitsAllowed(void);
    fprintf(stderr, "gnm: submits allowed: %d\n", sceGnmAreSubmitsAllowed());
    gnm.base = memory;
    gnm.image = gnm_alloc(sizeof gcn_image, 4096);
    memcpy(gnm.image, gcn_image, sizeof gcn_image);
    gnm.commands = gnm_alloc(COMMAND_WORDS * 4, 4096);
    gnm.arguments = gnm_alloc(ARGUMENT_BYTES, 4096);
    gnm.fence = &gnm.marks[0];
    gnm.zero = gnm_alloc(256, 256);
    memset(gnm.zero, 0, 256);
    gnm.failed = 0;
    if (!check()) { gnm.failed = 1; return 0; }
    gnm.ready = 1;
    if ((graphics_code = gnm_alloc(sizeof gcn_graphics, 4096))) memcpy(graphics_code, gcn_graphics, sizeof gcn_graphics);
    triangle_check();
    scene_check();
    return 1;
}
