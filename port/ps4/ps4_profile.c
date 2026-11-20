/* PS4 sampling profiler (port/ps4/PLAN.md): /data/dhd/profile (any file) turns it on. A sampler
   thread sends SIGPROF to the game's named threads (game, vu, gs, native) about every millisecond
   while a race is ridden (perf_log.c's riding); each thread's handler keeps the interrupted
   instruction's address. /data/dhd/profile.bin is rewritten every 10 s: a header (magic, the
   runtime address of ps4_profile_thread, to place the module), then one 64-bit record a sample:
   the thread's slot in the top byte, the address below. port/ps4/tools/profile_report.py names
   the addresses from the module's ELF.
   Written for this port; no third-party code. */
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define SLOTS 4
#define MAX_SAMPLES (1u << 20)          /* 8 MiB: about 4 minutes of race at 4 threads, 1 kHz */
#define FREEBSD_SA_SIGINFO 0x40         /* the console kernel's values (the header has Linux's) */
#define FREEBSD_SA_RESTART 0x02         /* the threads' waits resume after a sample */
#define UC_RIP 176                      /* ucontext_t: uc_sigmask (16), mcontext_t's mc_rip at 160 */

static const char *const names[SLOTS] = {"game", "vu", "gs", "native"};
static pthread_t threads[SLOTS];
static _Atomic int known[SLOTS];
static uint64_t *samples;
static _Atomic uint32_t count;
static int on = -1;

int perf_riding(void);  /* port/gs/perf_log.c */
void ps4_profile_thread(const char *name);

static void handler(int sig, void *info, void *context)
{
    (void)sig; (void)info;
    pthread_t self = pthread_self();
    unsigned slot = 0;
    while (slot < SLOTS && !(atomic_load(&known[slot]) && pthread_equal(threads[slot], self))) slot++;
    if (slot == SLOTS) return;
    uint64_t rip = *(uint64_t *)((char *)context + UC_RIP);
    uint32_t n = atomic_fetch_add(&count, 1);
    if (n < MAX_SAMPLES) samples[n] = ((uint64_t)slot << 56) | (rip & 0x00ffffffffffffffull);
}

static void save(void)
{
    uint32_t n = atomic_load(&count);
    if (n > MAX_SAMPLES) n = MAX_SAMPLES;
    FILE *f = fopen("/data/dhd/profile.tmp", "wb");
    if (!f) return;
    uint64_t header[2] = {0x31464f5250534844ull /* "DHSPROF1" */, (uint64_t)(uintptr_t)&ps4_profile_thread};
    fwrite(header, sizeof header, 1, f);
    fwrite(samples, sizeof *samples, n, f);
    fclose(f);
    rename("/data/dhd/profile.tmp", "/data/dhd/profile.bin");
}

static void *sampler(void *unused)
{
    (void)unused;
    unsigned ticks = 0;
    for (;;) {
        usleep(1000);
        if (perf_riding())
            for (unsigned i = 0; i < SLOTS; i++)
                if (atomic_load(&known[i])) pthread_kill(threads[i], SIGPROF);
        if (++ticks % 10000 == 0) save();  /* about every 10 s */
    }
    return NULL;
}

void ps4_profile_thread(const char *name)
{
    if (on < 0) {
        struct stat st;
        on = stat("/data/dhd/profile", &st) == 0;
        if (on) {
            samples = malloc(MAX_SAMPLES * sizeof *samples);
            struct sigaction sa;
            memset(&sa, 0, sizeof sa);
            sa.sa_sigaction = (void (*)(int, struct __siginfo *, void *))(void *)handler;
            sa.sa_flags = FREEBSD_SA_SIGINFO | FREEBSD_SA_RESTART;
            if (!samples || sigaction(SIGPROF, &sa, NULL) != 0) {
                fprintf(stderr, "ps4 profile: could not start\n");
                on = 0;
            } else {
                pthread_t t;
                pthread_create(&t, NULL, sampler, NULL);
                fprintf(stderr, "ps4 profile: sampling the game's threads in races into /data/dhd/profile.bin "
                        "(anchor %p)\n", (void *)&ps4_profile_thread);
            }
        }
    }
    if (!on) return;
    for (unsigned i = 0; i < SLOTS; i++)
        if (!strcmp(name, names[i]) && !atomic_load(&known[i])) {
            threads[i] = pthread_self();
            atomic_store(&known[i], 1);
        }
}
