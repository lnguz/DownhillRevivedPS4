/* The PS4's stand-ins for the Rust library (port/render: dhd_settings, dhd_disc, dhd_presence and
   dhd_render's PNG and statistics calls), which does not build for the console (port/ps4/PLAN.md).

   - dhd.ini: read and saved here from the C copy of the settings table (port/hle/settings_table.h,
     generated from the crate's table): each setting's variable from the file, else the table's
     built-in value.
   - Bindings: the game's default layouts (port/hle/controls_table.h); rebinding is not offered yet.
   - Online (ws_*, http_get), Discord presence, the disc's online check, texture packs' PNG files and
     the renderer statistics: not on the PS4 for now; each answers "unavailable" and the game carries
     on offline, as it does on a PC without a network. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "../hle/settings_table.h"
#include "../hle/controls_table.h"
#include "../render/include/dhd_settings.h"
#include "../render/include/dhd_disc.h"
#include "../render/include/dhd_presence.h"
#include "../net/ws.h"

/* ---- dhd.ini ---- */

static char *trim(char *s)
{
    while (*s == ' ' || *s == '\t') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) *--e = 0;
    return s;
}

/* The value the file gives key, into out. 1 when it has one. */
static int ini_value(FILE *f, const char *key, char *out, size_t size)
{
    char line[1024];
    rewind(f);
    while (fgets(line, sizeof line, f)) {
        char *p = trim(line);
        if (!*p || *p == '#' || *p == ';' || *p == '[') continue;
        char *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = 0;
        if (strcasecmp(trim(p), key)) continue;
        snprintf(out, size, "%s", trim(eq + 1));
        return 1;
    }
    return 0;
}

static const SettingInfo *setting_of(const char *key)
{
    for (size_t i = 0; i < sizeof setting_table / sizeof setting_table[0]; i++)
        if (!strcasecmp(setting_table[i].key, key)) return &setting_table[i];
    return NULL;
}

int dhd_settings_load(const char *path, DhdSettingsGet get, DhdSettingsSet set, DhdSettingsUnknown unknown)
{
    FILE *f = path ? fopen(path, "r") : NULL;
    char overrides[2048] = ",";
    for (size_t i = 0; i < sizeof setting_table / sizeof setting_table[0]; i++) {
        const SettingInfo *s = &setting_table[i];
        if (!s->env[0]) continue;
        if (get && get(s->env)) {  /* the environment overrides the file */
            if (strlen(overrides) + strlen(s->env) + 2 < sizeof overrides) {
                strcat(overrides, s->env);
                strcat(overrides, ",");
            }
            continue;
        }
        char value[512];
        if (!f || !ini_value(f, s->key, value, sizeof value))
            snprintf(value, sizeof value, "%s", s->builtin);
        if (value[0] && set) set(s->env, value);
    }
    if (set) set("DHD_ENV_OVERRIDES", overrides);
    if (f && unknown) {
        char line[1024];
        rewind(f);
        while (fgets(line, sizeof line, f)) {
            char *p = trim(line);
            if (!*p || *p == '#' || *p == ';' || *p == '[') continue;
            char *eq = strchr(p, '=');
            if (!eq) { unknown(p); continue; }
            *eq = 0;
            p = trim(p);
            if (!setting_of(p) && strcasecmp(p, "install_id") && strcasecmp(p, "iso") &&
                strcasecmp(p, "keyboard") && strncasecmp(p, "pad", 3))
                unknown(p);
        }
    }
    if (f) fclose(f);
    return f != NULL;
}

int dhd_settings_read(const char *path, const char *key, char *out, size_t size)
{
    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f) return 0;
    int have = ini_value(f, key, out, size);
    fclose(f);
    return have;
}

int dhd_settings_save(const char *path, const char *key, const char *value)
{
    if (!path || !key || !value) return 0;
    const char *disabled = getenv("DHD_SETTINGS");
    if (disabled && !strcasecmp(disabled, "none")) return 0;
    /* Read the old file, replace the key's first line and drop its others, write it back. */
    char *text = NULL;
    size_t length = 0;
    FILE *f = fopen(path, "r");
    if (f) {
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (n > 0 && n < (1 << 20)) {
            text = malloc((size_t)n + 1);
            if (text) {
                length = fread(text, 1, (size_t)n, f);
                text[length] = 0;
            }
        }
        fclose(f);
    }
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *out = fopen(tmp, "w");
    if (!out) { free(text); return 0; }
    if (!text) fputs("# Downhill Revived settings (PS4)\n", out);
    int written = 0;
    for (char *line = text, *next; line && *line; line = next) {
        next = strchr(line, '\n');
        if (next) *next++ = 0;
        char copy[1024];
        snprintf(copy, sizeof copy, "%s", line);
        char *p = trim(copy), *eq = strchr(p, '=');
        if (eq && *p != '#' && *p != ';') {
            *eq = 0;
            if (!strcasecmp(trim(p), key)) {
                if (!written) fprintf(out, "%s = %s\n", key, value);
                written = 1;
                continue;
            }
        }
        fprintf(out, "%s\n", line);
    }
    if (!written) fprintf(out, "%s = %s\n", key, value);
    free(text);
    int ok = fclose(out) == 0 && rename(tmp, path) == 0;
    if (!ok) remove(tmp);
    return ok;
}

int dhd_settings_install_id(const char *path, char *out, size_t size)
{
    if (size < 33) return 0;
    if (dhd_settings_read(path, "install_id", out, size) && strlen(out) == 32) return 1;
    static const char hex[] = "0123456789abcdef";
    srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)out);
    for (int i = 0; i < 32; i++) out[i] = hex[rand() & 15];
    out[32] = 0;
    dhd_settings_save(path, "install_id", out);
    return 1;
}

/* ---- bindings: the game's defaults ---- */

void dhd_controls_layouts(const char *const values[DHD_CONTROLS_DEVICES], DhdControlsLayout out[DHD_CONTROLS_DEVICES],
                          DhdSettingsUnknown problem)
{
    (void)values; (void)problem;
    for (int d = 0; d < DHD_CONTROLS_DEVICES; d++)
        memcpy(&out[d], &control_defaults[d], sizeof out[d]);
}

int dhd_controls_edit(const char *const values[DHD_CONTROLS_DEVICES], int device, int edit, int target, int slot, int input,
                      int kind, char *value, size_t value_size, char *message, size_t message_size)
{
    (void)values; (void)device; (void)edit; (void)target; (void)slot; (void)input; (void)kind;
    if (value && value_size) value[0] = 0;
    if (message && message_size) snprintf(message, message_size, "Rebinding is not available on PS4 yet");
    return CONTROL_REFUSED;
}

/* ---- not on the PS4 for now ---- */

int dhd_disc_check(const char *iso, const char *cache, uint8_t digest[32], char *note, size_t size,
                   const int *stop, uint32_t pause_us)
{
    (void)iso; (void)cache; (void)stop; (void)pause_us;
    memset(digest, 0, 32);
    if (note && size) snprintf(note, size, "not checked on PS4 (online races are off)");
    return -1;
}

int dhd_presence_start(const char *setting, DhdPresenceLog log) { (void)setting; (void)log; return 0; }
void dhd_presence_set(int kind, const char *track, int place, int riders, const char *room, int members,
                      int max_members)
{
    (void)kind; (void)track; (void)place; (void)riders; (void)room; (void)members; (void)max_members;
}
void dhd_presence_stop(void) {}

int dhdr_png_size(const uint8_t *data, size_t size, uint32_t *w, uint32_t *h)
{
    (void)data; (void)size; *w = *h = 0;
    return 0;
}
int dhdr_png_decode(const uint8_t *data, size_t size, uint8_t *rgba, uint32_t w, uint32_t h, uint32_t opaque)
{
    (void)data; (void)size; (void)rgba; (void)w; (void)h; (void)opaque;
    return 0;
}
uint32_t dhdr_perf_read(uint64_t *out, uint32_t count) { (void)out; (void)count; return 0; }
const char *dhdr_native_pipeline_crash(uint64_t *key) { (void)key; return NULL; }

WsConn *ws_start(const char *url, const WsHandler *h) { (void)url; (void)h; return NULL; }
void ws_send(WsConn *ws, int binary, const void *data, size_t n) { (void)ws; (void)binary; (void)data; (void)n; }
void ws_stop(WsConn *ws) { (void)ws; }
int ws_race_rtt_ms(WsConn *ws) { (void)ws; return -1; }
void ws_test_cut(int ms) { (void)ms; }
int http_get(const char *url, char *out, size_t cap) { (void)url; if (cap) out[0] = 0; return -1; }
