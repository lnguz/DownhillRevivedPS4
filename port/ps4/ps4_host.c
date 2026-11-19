/* PS4 host: the platform's start (port/ps4/PLAN.md). Runs before main (a constructor):

   - /data/dhd/ is the game's folder (host_exe_path): dhd.ini, saves, dhd.log. The package itself
     holds no game data; the player copies their own disc image there (any *.iso).
   - The log: stderr and stdout go to /data/dhd/dhd.log (the console has no terminal), the one
     before kept as dhd-previous.log, so a run can be read back over FTP.
   - DHD_ISO: the disc image found in /data/dhd/, unless dhd.ini or the environment names one.
   - Until the CPU renderer (phase 4) exists the game runs headless (DHD_HEADLESS), which also
     leaves the sound off; /data/dhd/picture (any file) skips that for renderer tests. */
#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include <orbis/UserService.h>
#include <orbis/libkernel.h>

#define DHD_DIR "/data/dhd"

static int exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static void find_disc(void)
{
    if (getenv("DHD_ISO")) return;
    DIR *dir = opendir(DHD_DIR);
    if (!dir) return;
    static char path[512];
    for (struct dirent *e; (e = readdir(dir));) {
        size_t n = strlen(e->d_name);
        if (n > 4 && !strcasecmp(e->d_name + n - 4, ".iso")) {
            snprintf(path, sizeof path, DHD_DIR "/%s", e->d_name);
            setenv("DHD_ISO", path, 1);
            break;
        }
    }
    closedir(dir);
}

static void ps4_start(void)
{
    mkdir("/data", 0777);
    mkdir(DHD_DIR, 0777);
    /* boot.txt: the first sign of life, without stdio, and how the log was set up */
    int boot = open(DHD_DIR "/boot.txt", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    char note[160];
    if (boot >= 0) write(boot, "ps4_start\n", 10);
    if (exists(DHD_DIR "/dhd.log")) rename(DHD_DIR "/dhd.log", DHD_DIR "/dhd-previous.log");
    /* The log: the console refuses dup2 onto descriptors 1 and 2, so stdio's stdout and stderr
       (the program's own musl, linked in) are pointed at the file instead: the descriptor in
       musl's FILE (offset 0x78, the field fileno() reads) becomes the log's. */
    int log = open(DHD_DIR "/dhd.log", O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0666);
    if (log >= 0) {
        fflush(stdout), fflush(stderr);
        *(int *)((char *)stderr + 0x78) = log;
        *(int *)((char *)stdout + 0x78) = log;
    }
    setvbuf(stderr, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    if (boot >= 0) {
        int n = snprintf(note, sizeof note, "log fd %d, stderr fd %d\n", log, fileno(stderr));
        write(boot, note, (size_t)n);
        close(boot);
    }
    write(2, "ps4: log open\n", 14);
    fprintf(stderr, "ps4: Downhill Revived starting in " DHD_DIR "\n");
    sceUserServiceInitialize(NULL);
    setenv("DHD_DATA_DIR", DHD_DIR, 0);
    find_disc();
    if (!exists(DHD_DIR "/picture")) setenv("DHD_HEADLESS", "1", 0);
    fprintf(stderr, "ps4: disc %s%s\n", getenv("DHD_ISO") ? getenv("DHD_ISO") : "(none in " DHD_DIR ")",
            getenv("DHD_HEADLESS") ? ", headless" : "");
}

/* The program's constructors (vif.c's publish_vu1, gpu_rust.c's and the host's own lines):
   crt1's start hands main to Sony's libc, which does not run the executable's .init_array, so
   main calls this first; a loader that did run them leaves ran set, and nothing runs twice. */
static int ran;
__attribute__((constructor(101))) static void constructors_ran(void) { ran = 1; }

void ps4_constructors(void)
{
    if (ran) {
        ps4_start();
        return;
    }
    typedef void (*Constructor)(void);
    extern Constructor __init_array_start[] __attribute__((visibility("hidden")));
    extern Constructor __init_array_end[] __attribute__((visibility("hidden")));
    ps4_start();
    for (Constructor *c = __init_array_start; c < __init_array_end; c++)
        if (*c && (uintptr_t)*c != (uintptr_t)-1 && *c != constructors_ran) (*c)();
    ran = 1;
}
