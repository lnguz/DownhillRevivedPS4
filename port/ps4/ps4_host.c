/* PS4 host: the platform's start (port/ps4/PLAN.md). Runs before main (a constructor):

   - /data/dhd/ is the game's folder (host_exe_path): dhd.ini, saves, dhd.log. The package itself
     holds no game data; the player copies their own disc image there (any *.iso).
   - The log: stderr and stdout go to /data/dhd/dhd.log (the console has no terminal), the one
     before kept as dhd-previous.log, so a run can be read back over FTP.
   - DHD_ISO: the disc image found in /data/dhd/, unless dhd.ini or the environment names one.
   - Until the CPU renderer (phase 4) exists the game runs headless (DHD_HEADLESS), which also
     leaves the sound off; /data/dhd/picture (any file) skips that for renderer tests. */
#include <dirent.h>
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

__attribute__((constructor(101))) static void ps4_start(void)
{
    mkdir("/data", 0777);
    mkdir(DHD_DIR, 0777);
    if (exists(DHD_DIR "/dhd.log")) rename(DHD_DIR "/dhd.log", DHD_DIR "/dhd-previous.log");
    if (freopen(DHD_DIR "/dhd.log", "w", stderr)) setvbuf(stderr, NULL, _IONBF, 0);
    if (freopen(DHD_DIR "/dhd.log", "a", stdout)) setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stderr, "ps4: Downhill Revived starting in " DHD_DIR "\n");
    sceUserServiceInitialize(NULL);
    setenv("DHD_DATA_DIR", DHD_DIR, 0);
    find_disc();
    if (!exists(DHD_DIR "/picture")) setenv("DHD_HEADLESS", "1", 0);
    fprintf(stderr, "ps4: disc %s%s\n", getenv("DHD_ISO") ? getenv("DHD_ISO") : "(none in " DHD_DIR ")",
            getenv("DHD_HEADLESS") ? ", headless" : "");
}
