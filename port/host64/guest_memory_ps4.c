/* PS4 mapping for guest_memory.c (port/ps4/PLAN.md). The default layout is the 8 GiB guard-page
   reservation (os_reserve, sceKernelReserveVirtualRange: address space only); the bounds-checked
   two-part layout (HOST64_TWO_PARTS, as iOS, finding 258) still builds: its parts are reserved at
   the places guest_memory.c picks from os_free_ranges (the process's gaps, read with
   sceKernelVirtualQuery: the console has no /proc). The 32 MiB RAM is one block of direct memory
   mapped at each of its views (sceKernelMapDirectMemory accepts the same physical block at several
   addresses, so the views share their contents as memfd's MAP_SHARED mappings do elsewhere).
   Committed ranges are direct memory too: a homebrew app's flexible memory budget is small, and
   the console's libc carries Linux's mmap constants for a FreeBSD kernel, so only Sony's calls
   are used here. No fault reporter: the system's crash handling ends a process that touches an
   uncommitted address, and host64_memory_wild reports before it aborts.
   Written for this port; no third-party code. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <orbis/libkernel.h>
#include "wasm-rt.h"
#include "core_host.h"
#include "guest_memory_os.h"

/* The layout: the 8 GiB guard-page reservation (os_reserve), the default since the per-access
   adapters of the two-part layout cost the EE's hot loops; the two-part one still builds
   (HOST64_TWO_PARTS=1) for comparison. */

#define RAM_SIZE 0x2000000ull
#define GUEST_SIZE 0x200000000ull            /* the 8 GiB layout's reservation */
#define DMEM_ALIGN 0x10000ull                /* 64 KiB: direct memory's allocation alignment */
#define MAP_FIXED_ 0x10                       /* SCE_KERNEL_MAP_FIXED */
#define MAP_NO_OVERWRITE_ 0x80                /* SCE_KERNEL_MAP_NO_OVERWRITE */
#define VQ_FIND_NEXT 1                        /* SCE_KERNEL_VQ_FIND_NEXT */
#define PROT_RW (ORBIS_KERNEL_PROT_CPU_READ | ORBIS_KERNEL_PROT_CPU_WRITE)
/* Where the guest's parts may go: above the first 4 GiB (the executable and the system's low
   mappings) and below the top of the user address space. */
#define USER_LOW 0x100000000ull
#define USER_TOP 0x7f0000000000ull

static off_t ram_dmem = -1;

/* Committed ranges: each is one direct memory allocation mapped at [at, at + size). Commits arrive
   for the module's memory, the scratchpad and hardware windows, the EE stacks and the bridge
   arena: a few hundred at most. */
typedef struct { uintptr_t at; uint64_t size; off_t dmem; } Commit;
static Commit *commits;
static unsigned commit_count, commit_cap;

void os_fail(const char *what) {
    fprintf(stderr, "host64 memory: %s (%d: %s)\n", what, errno, strerror(errno));
    {
        void host_log_drain(void) __attribute__((weak));
        if (host_log_drain) host_log_drain();
    }
    abort();
}

#if HOST64_TWO_PARTS
static int add_range(GuestRange **r, unsigned *n, unsigned *cap, uint64_t lo, uint64_t hi) {
    if (lo < USER_LOW) lo = USER_LOW;
    if (hi > USER_TOP) hi = USER_TOP;
    if (hi <= lo) return 1;
    if (*n == *cap) {
        unsigned grown = *cap ? *cap * 2 : 256;
        GuestRange *bigger = realloc(*r, grown * sizeof **r);
        if (!bigger) return 0;
        *r = bigger; *cap = grown;
    }
    (*r)[(*n)++] = (GuestRange){lo, hi};
    return 1;
}

/* The gaps between the process's mappings, as /proc/self/maps gives them on Linux. */
unsigned os_free_ranges(GuestRange **out) {
    GuestRange *r = NULL; unsigned n = 0, cap = 0;
    uint64_t prev = USER_LOW, at = USER_LOW;
    for (;;) {
        OrbisKernelVirtualQueryInfo info;
        memset(&info, 0, sizeof info);
        if (sceKernelVirtualQuery((const void *)(uintptr_t)at, VQ_FIND_NEXT, &info, sizeof info) != 0)
            break;
        uint64_t lo = (uint64_t)(uintptr_t)info.unk01, hi = (uint64_t)(uintptr_t)info.unk02;
        if (hi <= lo || lo >= USER_TOP) break;
        if (!add_range(&r, &n, &cap, prev, lo)) break;
        if (hi > prev) prev = hi;
        if (hi <= at) break;
        at = hi;
    }
    if (prev < USER_TOP) add_range(&r, &n, &cap, prev, USER_TOP);
    *out = r;
    return n;
}

int os_reserve_at(uintptr_t at, uint64_t size) {
    void *p = (void *)at;
    if (sceKernelReserveVirtualRange(&p, (size_t)size, MAP_FIXED_ | MAP_NO_OVERWRITE_, 0x4000) != 0)
        return 0;
    if ((uintptr_t)p != at) { sceKernelMunmap(p, (size_t)size); errno = EEXIST; return 0; }
    return 1;
}

#endif

int os_backing_open(void) {
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), RAM_SIZE, DMEM_ALIGN,
                                      ORBIS_KERNEL_WB_ONION, &ram_dmem) != 0) {
        ram_dmem = -1;
        return 0;
    }
    return 1;
}

#if !HOST64_TWO_PARTS
uintptr_t os_reserve(void) {
    if (!os_backing_open()) return 0;
    /* 8 GiB of address space, inaccessible: commits and the RAM views map over it (MAP_FIXED) */
    void *p = (void *)(uintptr_t)USER_LOW;
    if (sceKernelReserveVirtualRange(&p, (size_t)GUEST_SIZE, 0, DMEM_ALIGN) != 0) {
        os_backing_close();
        return 0;
    }
    fprintf(stderr, "host64 memory: 8 GiB guest reservation at %p\n", p);
    return (uintptr_t)p;
}
#endif

int os_split(uintptr_t at, uint64_t size) {
    (void)at; (void)size; return 1;
}

int os_map_ram(uintptr_t at, uint64_t size, uint64_t file_offset) {
    void *p = (void *)at;
    if (sceKernelMapDirectMemory(&p, (size_t)size, PROT_RW, MAP_FIXED_, ram_dmem + (off_t)file_offset,
                                 0x4000) != 0)
        return 0;
    return (uintptr_t)p == at;
}

int os_private(uintptr_t at, uint64_t size) {
    (void)at; (void)size; return 1;  /* already reserved by os_reserve(_at) */
}

static int committed(uintptr_t at) {
    for (unsigned i = 0; i < commit_count; i++)
        if (at >= commits[i].at && at < commits[i].at + commits[i].size) return 1;
    return 0;
}

/* Maps fresh, zeroed direct memory at [at, at + size), which must be uncommitted. */
static int commit_fresh(uintptr_t at, uint64_t size) {
    if (commit_count == commit_cap) {
        unsigned grown = commit_cap ? commit_cap * 2 : 64;
        Commit *bigger = realloc(commits, grown * sizeof *commits);
        if (!bigger) return 0;
        commits = bigger; commit_cap = grown;
    }
    uint64_t rounded = (size + DMEM_ALIGN - 1) & ~(DMEM_ALIGN - 1);
    off_t dmem;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), rounded, DMEM_ALIGN,
                                      ORBIS_KERNEL_WB_ONION, &dmem) != 0)
        return 0;
    void *p = (void *)at;
    if (sceKernelMapDirectMemory(&p, (size_t)size, PROT_RW, MAP_FIXED_, dmem, 0x4000) != 0 ||
        (uintptr_t)p != at) {
        sceKernelReleaseDirectMemory(dmem, rounded);
        return 0;
    }
    memset(p, 0, (size_t)size);  /* direct memory is not guaranteed zeroed */
    commits[commit_count++] = (Commit){at, size, dmem};
    return 1;
}

/* Commits [at, at + size) read/write; pages already committed keep their contents. */
int os_commit(uintptr_t at, uint64_t size) {
    const uint64_t page = 0x4000;
    uintptr_t end = at + size, run = 0;
    for (uintptr_t p = at; p < end; p += page) {
        int done = committed(p);
        if (!done && !run) run = p;
        if (done && run) { if (!commit_fresh(run, p - run)) return 0; run = 0; }
    }
    if (run && !commit_fresh(run, end - run)) return 0;
    return 1;
}

/* Returns [at, at + size) to the reserved state: the commits inside it are unmapped and released,
   and the range is reserved again. */
void os_decommit(const void *at, uint64_t size) {
    uintptr_t lo = (uintptr_t)at, hi = lo + size;
    for (unsigned i = 0; i < commit_count;) {
        Commit c = commits[i];
        if (c.at >= lo && c.at + c.size <= hi) {
            sceKernelMunmap((void *)c.at, (size_t)c.size);
            sceKernelReleaseDirectMemory(c.dmem, (c.size + DMEM_ALIGN - 1) & ~(DMEM_ALIGN - 1));
            commits[i] = commits[--commit_count];
        } else {
            i++;
        }
    }
    void *p = (void *)lo;
    sceKernelReserveVirtualRange(&p, (size_t)size, MAP_FIXED_, 0x4000);
}

int os_unmap_view(void *at, uint64_t size) {
    return sceKernelMunmap(at, (size_t)size) == 0;
}

int os_release(void *at, uint64_t size) {
    os_decommit(at, size);
    return sceKernelMunmap(at, (size_t)size) == 0;
}

void os_backing_close(void) {
    if (ram_dmem >= 0) sceKernelReleaseDirectMemory(ram_dmem, RAM_SIZE);
    ram_dmem = -1;
}

int os_fault_install(void) { return 1; }
void os_fault_remove(void) {}

/* An access outside the guest's two parts (memory_access.h, GUEST(), the bridge; finding 258). */
void host64_memory_wild(uint64_t address) {
    fprintf(stderr, "host64 memory: access of uncommitted guest address 0x%09llx\n",
            (unsigned long long)address);
    {
        void host_log_drain(void) __attribute__((weak));
        if (host_log_drain) host_log_drain();
    }
    abort();
}
