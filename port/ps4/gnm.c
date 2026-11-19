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

#include <stdio.h>
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
} gnm;

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
        uint32_t scratch;
        memcpy(&scratch, d + 4, 4);
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

void gnm_begin(void)
{
    gnm.count = 0;
    gnm.argument_used = 0;
    gnm.count += sceGnmDispatchInitDefaultHardwareState(gnm.commands, 256);
    acquire_mem();
    static const uint32_t start[3] = {0, 0, 0};
    set_sh(COMPUTE_START_X, start, 3);
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

int gnm_finish(void)
{
    /* the dispatches done, their writes in memory, then the fence */
    emit(PKT3(IT_EVENT_WRITE, 1, 0));  /* one body word: the event */
    emit(7u | (4u << 8));  /* CS_PARTIAL_FLUSH */
    acquire_mem();
    uint64_t fence = (uint64_t)(uintptr_t)gnm.fence;
    uint32_t value = ++gnm.fence_value;
    emit(PKT3(IT_EVENT_WRITE_EOP, 5, 0));
    emit(0x28u | (5u << 8));  /* BOTTOM_OF_PIPE_TS */
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
    const uint32_t *dcb[1] = {gnm.commands};
    uint32_t bytes[1] = {gnm.count * 4};
    int32_t rc = sceGnmSubmitCommandBuffers(1, dcb, bytes, NULL, NULL);
    if (gnm.fence_value == 1) fprintf(stderr, "gnm: step 4, submitted (%d), waiting for the fence\n", rc);
    if (rc != 0) {
        fprintf(stderr, "gnm: sceGnmSubmitCommandBuffers failed (0x%08x)\n", (unsigned)rc);
        return 0;
    }
    uint64_t start = sceKernelGetProcessTime();
    while (*gnm.fence != value) {
        if (sceKernelGetProcessTime() - start > 3000000) {
            fprintf(stderr, "gnm: the GPU did not finish within 3 s (fence %u, want %u; start mark %u)\n",
                    *gnm.fence, value, gnm.marks[1]);
            gnm.failed = 1;
            return 0;
        }
        sceKernelUsleep(20);
    }
    return 1;
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
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), 2u << 20, 2u << 20, 3, &marks_offset) != 0 ||
        sceKernelMapDirectMemory(&marks, 2u << 20, 0x33, 0, marks_offset, 2u << 20) != 0) {
        fprintf(stderr, "gnm: no garlic memory\n");
        return 0;
    }
    gnm.marks = marks;
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
    return 1;
}
