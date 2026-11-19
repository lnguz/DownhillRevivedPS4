/* The subset of SDL3 the host uses (port/host, port/hle, port/gs), over the PS4's own libraries
   (port/ps4/PLAN.md). SDL3's headers (the release sdl3.py pins) give the types and declarations;
   this file is the implementation, written for this port. SDL has no PS4 platform, and the host
   only needs threads, locks, atomics, time, properties, an event queue, one full-screen window,
   one audio stream and the DualShock 4.

   - Threads and locks: pthreads. Atomics: the compiler's builtins.
   - Time: sceKernelGetProcessTime (microseconds since the process started).
   - Audio: one thread calls the stream's get-callback for each period and plays it with
     sceAudioOut, whose output call paces the thread, as SDL's device thread does.
   - Controller: the initial user's DualShock 4 (scePad) as one SDL gamepad of type PS4, id 1.
   - Files: the game's folder (SDL_GetBasePath) is /data/dhd/: dhd.ini, dhd.log, saves.
   - Window and display: one 1920x1080 full-screen window; nothing to move or resize. */
#define _GNU_SOURCE  /* pthread_setname_np */
#include <SDL3/SDL.h>

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <orbis/AudioOut.h>
#include <orbis/Pad.h>
#include <orbis/Sysmodule.h>
#include <orbis/UserService.h>
#include <orbis/libkernel.h>

#define PS4_BASE_PATH "/data/dhd/"
#define DISPLAY_W 1920
#define DISPLAY_H 1080

static char error_text[256] = "";

static bool fail(const char *what)
{
    snprintf(error_text, sizeof error_text, "%s", what);
    return false;
}

const char *SDL_GetError(void) { return error_text; }

/* ---------------- atomics and spinlocks ---------------- */

int SDL_GetAtomicInt(SDL_AtomicInt *a) { return __atomic_load_n(&a->value, __ATOMIC_SEQ_CST); }
int SDL_SetAtomicInt(SDL_AtomicInt *a, int v) { return __atomic_exchange_n(&a->value, v, __ATOMIC_SEQ_CST); }
int SDL_AddAtomicInt(SDL_AtomicInt *a, int v) { return __atomic_fetch_add(&a->value, v, __ATOMIC_SEQ_CST); }
bool SDL_CompareAndSwapAtomicInt(SDL_AtomicInt *a, int oldval, int newval)
{
    return __atomic_compare_exchange_n(&a->value, &oldval, newval, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
void SDL_LockSpinlock(SDL_SpinLock *lock)
{
    while (__atomic_exchange_n(lock, 1, __ATOMIC_ACQUIRE))
        while (__atomic_load_n(lock, __ATOMIC_RELAXED)) __builtin_ia32_pause();
}
void SDL_UnlockSpinlock(SDL_SpinLock *lock) { __atomic_store_n(lock, 0, __ATOMIC_RELEASE); }

/* ---------------- mutexes, conditions, semaphores ---------------- */

struct SDL_Mutex { pthread_mutex_t m; };
struct SDL_Condition { pthread_cond_t c; };
struct SDL_Semaphore { pthread_mutex_t m; pthread_cond_t c; Uint32 count; };

SDL_Mutex *SDL_CreateMutex(void)
{
    SDL_Mutex *m = malloc(sizeof *m);
    if (!m) return NULL;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);  /* SDL's mutexes are recursive */
    pthread_mutex_init(&m->m, &attr);
    pthread_mutexattr_destroy(&attr);
    return m;
}
void SDL_LockMutex(SDL_Mutex *m) { if (m) pthread_mutex_lock(&m->m); }
void SDL_UnlockMutex(SDL_Mutex *m) { if (m) pthread_mutex_unlock(&m->m); }
void SDL_DestroyMutex(SDL_Mutex *m) { if (m) { pthread_mutex_destroy(&m->m); free(m); } }

SDL_Condition *SDL_CreateCondition(void)
{
    SDL_Condition *c = malloc(sizeof *c);
    if (c) pthread_cond_init(&c->c, NULL);
    return c;
}
void SDL_DestroyCondition(SDL_Condition *c) { if (c) { pthread_cond_destroy(&c->c); free(c); } }
void SDL_SignalCondition(SDL_Condition *c) { if (c) pthread_cond_signal(&c->c); }
void SDL_BroadcastCondition(SDL_Condition *c) { if (c) pthread_cond_broadcast(&c->c); }

static void deadline_in(struct timespec *ts, Sint64 ns)
{
    clock_gettime(CLOCK_REALTIME, ts);
    ts->tv_sec += (time_t)(ns / 1000000000);
    ts->tv_nsec += (long)(ns % 1000000000);
    if (ts->tv_nsec >= 1000000000) { ts->tv_sec++; ts->tv_nsec -= 1000000000; }
}

bool SDL_WaitConditionTimeoutNS(SDL_Condition *c, SDL_Mutex *m, Sint64 ns)
{
    if (!c || !m) return false;
    if (ns < 0) return pthread_cond_wait(&c->c, &m->m) == 0;
    struct timespec ts;
    deadline_in(&ts, ns);
    return pthread_cond_timedwait(&c->c, &m->m, &ts) == 0;
}
void SDL_WaitCondition(SDL_Condition *c, SDL_Mutex *m) { SDL_WaitConditionTimeoutNS(c, m, -1); }
bool SDL_WaitConditionTimeout(SDL_Condition *c, SDL_Mutex *m, Sint32 ms)
{
    return SDL_WaitConditionTimeoutNS(c, m, ms < 0 ? -1 : (Sint64)ms * 1000000);
}

SDL_Semaphore *SDL_CreateSemaphore(Uint32 initial)
{
    SDL_Semaphore *s = malloc(sizeof *s);
    if (!s) return NULL;
    pthread_mutex_init(&s->m, NULL);
    pthread_cond_init(&s->c, NULL);
    s->count = initial;
    return s;
}
void SDL_DestroySemaphore(SDL_Semaphore *s)
{
    if (!s) return;
    pthread_cond_destroy(&s->c);
    pthread_mutex_destroy(&s->m);
    free(s);
}
bool SDL_WaitSemaphoreTimeoutNS(SDL_Semaphore *s, Sint64 ns)
{
    if (!s) return false;
    bool got = true;
    pthread_mutex_lock(&s->m);
    if (ns < 0) {
        while (!s->count) pthread_cond_wait(&s->c, &s->m);
    } else {
        struct timespec ts;
        deadline_in(&ts, ns);
        while (!s->count)
            if (pthread_cond_timedwait(&s->c, &s->m, &ts) != 0) { got = s->count > 0; break; }
    }
    if (got && s->count) s->count--;
    else got = false;
    pthread_mutex_unlock(&s->m);
    return got;
}
void SDL_WaitSemaphore(SDL_Semaphore *s) { SDL_WaitSemaphoreTimeoutNS(s, -1); }
bool SDL_WaitSemaphoreTimeout(SDL_Semaphore *s, Sint32 ms)
{
    return SDL_WaitSemaphoreTimeoutNS(s, ms < 0 ? -1 : (Sint64)ms * 1000000);
}
void SDL_SignalSemaphore(SDL_Semaphore *s)
{
    if (!s) return;
    pthread_mutex_lock(&s->m);
    s->count++;
    pthread_cond_signal(&s->c);
    pthread_mutex_unlock(&s->m);
}
Uint32 SDL_GetSemaphoreValue(SDL_Semaphore *s)
{
    if (!s) return 0;
    pthread_mutex_lock(&s->m);
    Uint32 v = s->count;
    pthread_mutex_unlock(&s->m);
    return v;
}

/* ---------------- properties ---------------- */

typedef struct Property {
    char *name;
    SDL_PropertyType type;
    void *pointer;
    char *string;
    Sint64 number;
    float floating;
    bool boolean;
    struct Property *next;
} Property;
typedef struct { SDL_PropertiesID id; Property *first; } PropertySet;

static pthread_mutex_t properties_lock = PTHREAD_MUTEX_INITIALIZER;
static PropertySet *property_sets;
static unsigned property_set_count;
static SDL_PropertiesID next_properties = 1;

static PropertySet *properties_of(SDL_PropertiesID id)
{
    for (unsigned i = 0; i < property_set_count; i++)
        if (property_sets[i].id == id) return &property_sets[i];
    return NULL;
}

SDL_PropertiesID SDL_CreateProperties(void)
{
    pthread_mutex_lock(&properties_lock);
    PropertySet *grown = realloc(property_sets, (property_set_count + 1) * sizeof *grown);
    SDL_PropertiesID id = 0;
    if (grown) {
        property_sets = grown;
        id = next_properties++;
        property_sets[property_set_count++] = (PropertySet){id, NULL};
    }
    pthread_mutex_unlock(&properties_lock);
    return id;
}

void SDL_DestroyProperties(SDL_PropertiesID id)
{
    pthread_mutex_lock(&properties_lock);
    for (unsigned i = 0; i < property_set_count; i++) {
        if (property_sets[i].id != id) continue;
        for (Property *p = property_sets[i].first, *next; p; p = next) {
            next = p->next;
            free(p->name);
            free(p->string);
            free(p);
        }
        property_sets[i] = property_sets[--property_set_count];
        break;
    }
    pthread_mutex_unlock(&properties_lock);
}

/* The property name in set id, made when missing; the caller holds properties_lock. */
static Property *property(SDL_PropertiesID id, const char *name, bool make)
{
    PropertySet *set = properties_of(id);
    if (!set || !name) return NULL;
    for (Property *p = set->first; p; p = p->next)
        if (!strcmp(p->name, name)) return p;
    if (!make) return NULL;
    Property *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->name = strdup(name);
    p->next = set->first;
    set->first = p;
    return p;
}

bool SDL_SetPointerProperty(SDL_PropertiesID id, const char *name, void *value)
{
    pthread_mutex_lock(&properties_lock);
    Property *p = property(id, name, true);
    if (p) { p->type = SDL_PROPERTY_TYPE_POINTER; p->pointer = value; }
    pthread_mutex_unlock(&properties_lock);
    return p != NULL;
}
bool SDL_SetStringProperty(SDL_PropertiesID id, const char *name, const char *value)
{
    pthread_mutex_lock(&properties_lock);
    Property *p = property(id, name, true);
    if (p) { p->type = SDL_PROPERTY_TYPE_STRING; free(p->string); p->string = value ? strdup(value) : NULL; }
    pthread_mutex_unlock(&properties_lock);
    return p != NULL;
}
bool SDL_SetNumberProperty(SDL_PropertiesID id, const char *name, Sint64 value)
{
    pthread_mutex_lock(&properties_lock);
    Property *p = property(id, name, true);
    if (p) { p->type = SDL_PROPERTY_TYPE_NUMBER; p->number = value; }
    pthread_mutex_unlock(&properties_lock);
    return p != NULL;
}
bool SDL_SetBooleanProperty(SDL_PropertiesID id, const char *name, bool value)
{
    pthread_mutex_lock(&properties_lock);
    Property *p = property(id, name, true);
    if (p) { p->type = SDL_PROPERTY_TYPE_BOOLEAN; p->boolean = value; }
    pthread_mutex_unlock(&properties_lock);
    return p != NULL;
}
void *SDL_GetPointerProperty(SDL_PropertiesID id, const char *name, void *fallback)
{
    pthread_mutex_lock(&properties_lock);
    Property *p = property(id, name, false);
    void *v = p && p->type == SDL_PROPERTY_TYPE_POINTER ? p->pointer : fallback;
    pthread_mutex_unlock(&properties_lock);
    return v;
}
Sint64 SDL_GetNumberProperty(SDL_PropertiesID id, const char *name, Sint64 fallback)
{
    pthread_mutex_lock(&properties_lock);
    Property *p = property(id, name, false);
    Sint64 v = p && p->type == SDL_PROPERTY_TYPE_NUMBER ? p->number : fallback;
    pthread_mutex_unlock(&properties_lock);
    return v;
}
bool SDL_GetBooleanProperty(SDL_PropertiesID id, const char *name, bool fallback)
{
    pthread_mutex_lock(&properties_lock);
    Property *p = property(id, name, false);
    bool v = p && p->type == SDL_PROPERTY_TYPE_BOOLEAN ? p->boolean : fallback;
    pthread_mutex_unlock(&properties_lock);
    return v;
}

/* ---------------- threads and thread-local storage ---------------- */

struct SDL_Thread {
    pthread_t handle;
    SDL_ThreadFunction fn;
    void *data;
    char name[32];
    int status;
};

static void *thread_start(void *arg)
{
    SDL_Thread *t = arg;
    if (t->name[0]) pthread_setname_np(pthread_self(), t->name);
    t->status = t->fn(t->data);
    return NULL;
}

static SDL_Thread *thread_create(SDL_ThreadFunction fn, const char *name, void *data, size_t stack)
{
    SDL_Thread *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->fn = fn;
    t->data = data;
    snprintf(t->name, sizeof t->name, "%s", name ? name : "");
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    /* SDL's default when nothing is asked: the platform's, which on the PS4 is small for the game's
       threads; 1 MiB unless the caller names a size */
    pthread_attr_setstacksize(&attr, stack ? stack : 1024 * 1024);
    int rc = pthread_create(&t->handle, &attr, thread_start, t);
    pthread_attr_destroy(&attr);
    if (rc) { free(t); fail("pthread_create failed"); return NULL; }
    return t;
}

SDL_Thread *SDL_CreateThreadRuntime(SDL_ThreadFunction fn, const char *name, void *data,
                                    SDL_FunctionPointer begin, SDL_FunctionPointer end)
{
    (void)begin; (void)end;
    return thread_create(fn, name, data, 0);
}

SDL_Thread *SDL_CreateThreadWithPropertiesRuntime(SDL_PropertiesID props, SDL_FunctionPointer begin,
                                                  SDL_FunctionPointer end)
{
    (void)begin; (void)end;
    SDL_ThreadFunction fn = (SDL_ThreadFunction)SDL_GetPointerProperty(props, SDL_PROP_THREAD_CREATE_ENTRY_FUNCTION_POINTER, NULL);
    pthread_mutex_lock(&properties_lock);
    Property *p = property(props, SDL_PROP_THREAD_CREATE_NAME_STRING, false);
    char name[32] = "";
    if (p && p->type == SDL_PROPERTY_TYPE_STRING && p->string) snprintf(name, sizeof name, "%s", p->string);
    pthread_mutex_unlock(&properties_lock);
    void *data = SDL_GetPointerProperty(props, SDL_PROP_THREAD_CREATE_USERDATA_POINTER, NULL);
    Sint64 stack = SDL_GetNumberProperty(props, SDL_PROP_THREAD_CREATE_STACKSIZE_NUMBER, 0);
    if (!fn) { fail("no thread function"); return NULL; }
    return thread_create(fn, name, data, stack > 0 ? (size_t)stack : 0);
}

void SDL_WaitThread(SDL_Thread *t, int *status)
{
    if (!t) return;
    pthread_join(t->handle, NULL);
    if (status) *status = t->status;
    free(t);
}

void SDL_DetachThread(SDL_Thread *t)
{
    if (!t) return;
    pthread_detach(t->handle);  /* the record stays: the thread may still read it */
}

SDL_ThreadID SDL_GetCurrentThreadID(void) { return (SDL_ThreadID)(uintptr_t)pthread_self(); }

bool SDL_SetCurrentThreadPriority(SDL_ThreadPriority priority)
{
    (void)priority;  /* the console schedules the game's threads itself */
    return true;
}

bool SDL_SetTLS(SDL_TLSID *id, const void *value, SDL_TLSDestructorCallback destructor)
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    if (!SDL_GetAtomicInt(id)) {
        pthread_mutex_lock(&lock);
        if (!SDL_GetAtomicInt(id)) {
            pthread_key_t key;
            if (pthread_key_create(&key, destructor)) { pthread_mutex_unlock(&lock); return fail("TLS"); }
            SDL_SetAtomicInt(id, (int)key + 1);
        }
        pthread_mutex_unlock(&lock);
    }
    return pthread_setspecific((pthread_key_t)(SDL_GetAtomicInt(id) - 1), value) == 0;
}

void *SDL_GetTLS(SDL_TLSID *id)
{
    int v = SDL_GetAtomicInt(id);
    return v ? pthread_getspecific((pthread_key_t)(v - 1)) : NULL;
}

bool SDL_ShouldInit(SDL_InitState *state)
{
    for (;;) {
        if (SDL_CompareAndSwapAtomicInt(&state->status, 0, 1)) {  /* uninitialized -> initializing */
            state->thread = SDL_GetCurrentThreadID();
            return true;
        }
        int status = SDL_GetAtomicInt(&state->status);
        if (status == 2) return false;  /* initialized */
        if (status == 1 && state->thread == SDL_GetCurrentThreadID()) return false;
        SDL_Delay(1);
    }
}

void SDL_SetInitialized(SDL_InitState *state, bool initialized)
{
    state->thread = 0;
    SDL_SetAtomicInt(&state->status, initialized ? 2 : 0);
}

/* ---------------- time ---------------- */

Uint64 SDL_GetTicksNS(void) { return (Uint64)sceKernelGetProcessTime() * 1000; }
Uint64 SDL_GetTicks(void) { return (Uint64)sceKernelGetProcessTime() / 1000; }
Uint64 SDL_GetPerformanceCounter(void) { return (Uint64)sceKernelGetProcessTime(); }
Uint64 SDL_GetPerformanceFrequency(void) { return 1000000; }
void SDL_DelayNS(Uint64 ns) { if (ns) sceKernelUsleep((unsigned)((ns + 999) / 1000)); }
void SDL_Delay(Uint32 ms) { SDL_DelayNS((Uint64)ms * 1000000); }

bool SDL_GetCurrentTime(SDL_Time *ticks)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts)) return fail("clock_gettime");
    *ticks = (SDL_Time)ts.tv_sec * 1000000000 + ts.tv_nsec;
    return true;
}

bool SDL_TimeToDateTime(SDL_Time ticks, SDL_DateTime *dt, bool localTime)
{
    time_t seconds = (time_t)(ticks / 1000000000);
    struct tm tm;
    if (!(localTime ? localtime_r(&seconds, &tm) : gmtime_r(&seconds, &tm))) return fail("time");
    dt->year = tm.tm_year + 1900;
    dt->month = tm.tm_mon + 1;
    dt->day = tm.tm_mday;
    dt->hour = tm.tm_hour;
    dt->minute = tm.tm_min;
    dt->second = tm.tm_sec;
    dt->nanosecond = (int)(ticks % 1000000000);
    dt->day_of_week = tm.tm_wday;
    dt->utc_offset = 0;
    return true;
}

/* ---------------- init and hints ---------------- */

static SDL_InitFlags initialized;
static void pad_open(void);

bool SDL_InitSubSystem(SDL_InitFlags flags)
{
    if (flags & (SDL_INIT_GAMEPAD | SDL_INIT_JOYSTICK)) pad_open();
    initialized |= flags;
    return true;
}
bool SDL_Init(SDL_InitFlags flags) { return SDL_InitSubSystem(flags); }
SDL_InitFlags SDL_WasInit(SDL_InitFlags flags) { return flags ? initialized & flags : initialized; }
bool SDL_SetHint(const char *name, const char *value) { (void)name; (void)value; return true; }
const char *SDL_GetCurrentVideoDriver(void) { return "ps4"; }
const char *SDL_GetCurrentAudioDriver(void) { return "ps4"; }

/* ---------------- events ---------------- */

#define EVENT_QUEUE 256
static SDL_Event events[EVENT_QUEUE];
static unsigned event_head, event_count;
static pthread_mutex_t event_lock = PTHREAD_MUTEX_INITIALIZER;
static struct { SDL_EventFilter fn; void *userdata; } watches[8];
static int watch_count;
static Uint32 next_user_event = SDL_EVENT_USER;
static void pad_events(void);

bool SDL_PushEvent(SDL_Event *e)
{
    for (int i = 0; i < watch_count; i++) watches[i].fn(watches[i].userdata, e);
    pthread_mutex_lock(&event_lock);
    bool ok = event_count < EVENT_QUEUE;
    if (ok) events[(event_head + event_count++) % EVENT_QUEUE] = *e;
    pthread_mutex_unlock(&event_lock);
    return ok;
}

bool SDL_PollEvent(SDL_Event *e)
{
    pad_events();
    pthread_mutex_lock(&event_lock);
    bool have = event_count > 0;
    if (have) {
        if (e) {
            *e = events[event_head];
            event_head = (event_head + 1) % EVENT_QUEUE;
            event_count--;
        }
    }
    pthread_mutex_unlock(&event_lock);
    return have;
}

bool SDL_WaitEventTimeout(SDL_Event *e, Sint32 ms)
{
    Uint64 end = SDL_GetTicks() + (ms < 0 ? 0 : (Uint64)ms);
    for (;;) {
        if (SDL_PollEvent(e)) return true;
        if (ms >= 0 && SDL_GetTicks() >= end) return false;
        SDL_Delay(1);
    }
}

bool SDL_AddEventWatch(SDL_EventFilter filter, void *userdata)
{
    if (watch_count == 8) return fail("too many event watches");
    watches[watch_count].fn = filter;
    watches[watch_count++].userdata = userdata;
    return true;
}

Uint32 SDL_RegisterEvents(int count)
{
    if (count <= 0) return 0;
    Uint32 first = next_user_event;
    next_user_event += (Uint32)count;
    return first;
}

/* ---------------- window and display: one full-screen 1920x1080 window ---------------- */

struct SDL_Window { SDL_PropertiesID props; };
static SDL_Window the_window;
static bool window_made;
static const SDL_DisplayMode display_mode = {1, SDL_PIXELFORMAT_XRGB8888, DISPLAY_W, DISPLAY_H, 1.0f, 60.0f, 60, 1, NULL};

SDL_Window *SDL_CreateWindow(const char *title, int w, int h, SDL_WindowFlags flags)
{
    (void)title; (void)w; (void)h; (void)flags;
    if (!window_made) { the_window.props = SDL_CreateProperties(); window_made = true; }
    return &the_window;
}
SDL_Window *SDL_GetWindowFromID(SDL_WindowID id) { return id == 1 && window_made ? &the_window : NULL; }
SDL_WindowID SDL_GetWindowID(SDL_Window *window) { return window ? 1 : 0; }
SDL_Window **SDL_GetWindows(int *count)
{
    SDL_Window **list = calloc(2, sizeof *list);
    if (list && window_made) list[0] = &the_window;
    if (count) *count = list && window_made ? 1 : 0;
    return list;
}
SDL_PropertiesID SDL_GetWindowProperties(SDL_Window *window) { return window ? window->props : 0; }
SDL_WindowFlags SDL_GetWindowFlags(SDL_Window *window) { (void)window; return SDL_WINDOW_FULLSCREEN | SDL_WINDOW_INPUT_FOCUS; }
bool SDL_GetWindowSize(SDL_Window *window, int *w, int *h)
{
    (void)window; if (w) *w = DISPLAY_W; if (h) *h = DISPLAY_H; return true;
}
bool SDL_GetWindowSizeInPixels(SDL_Window *window, int *w, int *h) { return SDL_GetWindowSize(window, w, h); }
bool SDL_GetWindowBordersSize(SDL_Window *window, int *top, int *left, int *bottom, int *right)
{
    (void)window;
    if (top) *top = 0; if (left) *left = 0; if (bottom) *bottom = 0; if (right) *right = 0;
    return true;
}
float SDL_GetWindowDisplayScale(SDL_Window *window) { (void)window; return 1.0f; }
float SDL_GetWindowPixelDensity(SDL_Window *window) { (void)window; return 1.0f; }
SDL_DisplayID SDL_GetDisplayForWindow(SDL_Window *window) { (void)window; return 1; }
SDL_DisplayID SDL_GetPrimaryDisplay(void) { return 1; }
bool SDL_GetDisplayBounds(SDL_DisplayID id, SDL_Rect *rect)
{
    (void)id; rect->x = rect->y = 0; rect->w = DISPLAY_W; rect->h = DISPLAY_H; return true;
}
bool SDL_GetDisplayUsableBounds(SDL_DisplayID id, SDL_Rect *rect) { return SDL_GetDisplayBounds(id, rect); }
const SDL_DisplayMode *SDL_GetDesktopDisplayMode(SDL_DisplayID id) { (void)id; return &display_mode; }
const SDL_DisplayMode *SDL_GetCurrentDisplayMode(SDL_DisplayID id) { (void)id; return &display_mode; }
SDL_DisplayMode **SDL_GetFullscreenDisplayModes(SDL_DisplayID id, int *count)
{
    (void)id;
    SDL_DisplayMode **list = calloc(1, 2 * sizeof *list + sizeof display_mode);
    if (!list) { if (count) *count = 0; return NULL; }
    SDL_DisplayMode *mode = (SDL_DisplayMode *)(list + 2);
    *mode = display_mode;
    list[0] = mode;
    if (count) *count = 1;
    return list;
}
bool SDL_ShowWindow(SDL_Window *window) { (void)window; return true; }
bool SDL_HideWindow(SDL_Window *window) { (void)window; return true; }
bool SDL_RaiseWindow(SDL_Window *window) { (void)window; return true; }
bool SDL_SyncWindow(SDL_Window *window) { (void)window; return true; }
bool SDL_SetWindowBordered(SDL_Window *window, bool bordered) { (void)window; (void)bordered; return true; }
bool SDL_SetWindowFullscreen(SDL_Window *window, bool fullscreen) { (void)window; (void)fullscreen; return true; }
bool SDL_SetWindowIcon(SDL_Window *window, SDL_Surface *icon) { (void)window; (void)icon; return true; }
bool SDL_SetWindowPosition(SDL_Window *window, int x, int y) { (void)window; (void)x; (void)y; return true; }
bool SDL_SetWindowSize(SDL_Window *window, int w, int h) { (void)window; (void)w; (void)h; return true; }
bool SDL_SetWindowTitle(SDL_Window *window, const char *title) { (void)window; (void)title; return true; }
bool SDL_ShowCursor(void) { return true; }
bool SDL_HideCursor(void) { return true; }
bool SDL_StartTextInput(SDL_Window *window) { (void)window; return true; }
bool SDL_StopTextInput(SDL_Window *window) { (void)window; return true; }
bool SDL_TextInputActive(SDL_Window *window) { (void)window; return false; }
char *SDL_GetClipboardText(void) { return strdup(""); }
bool SDL_ShowSimpleMessageBox(SDL_MessageBoxFlags flags, const char *title, const char *message, SDL_Window *window)
{
    (void)flags; (void)window;
    fprintf(stderr, "message: %s: %s\n", title ? title : "", message ? message : "");
    return true;
}

/* ---------------- audio: one stream, its get-callback, sceAudioOut ---------------- */

#define AUDIO_RATE 48000
#define AUDIO_PERIOD 512             /* frames per sceAudioOut call, about 10.7 ms */
#define AUDIO_QUEUE (AUDIO_RATE / 2) /* the stream's buffer: half a second of stereo frames */
#define AUDIO_OUT_ALREADY_INIT 0x8026000Eu

struct SDL_AudioStream {
    SDL_AudioStreamCallback callback;
    void *userdata;
    int16_t queue[AUDIO_QUEUE * 2];
    unsigned read, count;            /* in frames */
    pthread_mutex_t lock;
    pthread_t thread;
    volatile int running, paused;
    float gain;
    int handle;
};

static void *audio_thread(void *arg)
{
    SDL_AudioStream *s = arg;
    static int16_t out[AUDIO_PERIOD * 2];
    while (s->running) {
        if (s->paused) {
            memset(out, 0, sizeof out);
        } else {
            unsigned queued;
            pthread_mutex_lock(&s->lock);
            queued = s->count;
            pthread_mutex_unlock(&s->lock);
            if (queued < AUDIO_PERIOD && s->callback) {
                int need = (int)(AUDIO_PERIOD - queued) * 4;
                s->callback(s->userdata, s, need, need);
            }
            pthread_mutex_lock(&s->lock);
            unsigned take = s->count < AUDIO_PERIOD ? s->count : AUDIO_PERIOD;
            for (unsigned i = 0; i < take; i++) {
                unsigned at = (s->read + i) % AUDIO_QUEUE;
                out[i * 2] = s->queue[at * 2];
                out[i * 2 + 1] = s->queue[at * 2 + 1];
            }
            s->read = (s->read + take) % AUDIO_QUEUE;
            s->count -= take;
            pthread_mutex_unlock(&s->lock);
            if (take < AUDIO_PERIOD) memset(out + take * 2, 0, (AUDIO_PERIOD - take) * 4);
            if (s->gain != 1.0f)
                for (unsigned i = 0; i < AUDIO_PERIOD * 2; i++) out[i] = (int16_t)(out[i] * s->gain);
        }
        sceAudioOutOutput(s->handle, out);  /* paces the thread: returns as the last block plays */
    }
    return NULL;
}

SDL_AudioStream *SDL_OpenAudioDeviceStream(SDL_AudioDeviceID devid, const SDL_AudioSpec *spec,
                                           SDL_AudioStreamCallback callback, void *userdata)
{
    (void)devid;
    if (!spec || spec->format != SDL_AUDIO_S16 || spec->channels != 2 || spec->freq != AUDIO_RATE) {
        fail("the PS4 audio stream plays 48000 Hz stereo S16");
        return NULL;
    }
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_AUDIOOUT);
    int rc = sceAudioOutInit();
    if (rc != 0 && (unsigned)rc != AUDIO_OUT_ALREADY_INIT) { fail("sceAudioOutInit failed"); return NULL; }
    int user = ORBIS_USER_SERVICE_USER_ID_SYSTEM, initial = -1;
    if (sceUserServiceGetInitialUser(&initial) == 0 && initial > 0) user = initial;
    int handle = sceAudioOutOpen(user, ORBIS_AUDIO_OUT_PORT_TYPE_MAIN, 0, AUDIO_PERIOD, AUDIO_RATE,
                                 ORBIS_AUDIO_OUT_PARAM_FORMAT_S16_STEREO);
    if (handle <= 0)
        handle = sceAudioOutOpen(ORBIS_USER_SERVICE_USER_ID_SYSTEM, ORBIS_AUDIO_OUT_PORT_TYPE_MAIN, 0, AUDIO_PERIOD,
                                 AUDIO_RATE, ORBIS_AUDIO_OUT_PARAM_FORMAT_S16_STEREO);
    if (handle <= 0) { fail("sceAudioOutOpen failed"); return NULL; }
    SDL_AudioStream *s = calloc(1, sizeof *s);
    if (!s) { sceAudioOutClose(handle); return NULL; }
    s->callback = callback;
    s->userdata = userdata;
    s->handle = handle;
    s->gain = 1.0f;
    s->paused = 1;  /* SDL opens the device paused */
    s->running = 1;
    pthread_mutex_init(&s->lock, NULL);
    if (pthread_create(&s->thread, NULL, audio_thread, s)) {
        sceAudioOutClose(handle);
        free(s);
        fail("audio thread");
        return NULL;
    }
    return s;
}

bool SDL_PutAudioStreamData(SDL_AudioStream *s, const void *buf, int len)
{
    if (!s || len < 0) return false;
    const int16_t *in = buf;
    unsigned frames = (unsigned)len / 4;
    pthread_mutex_lock(&s->lock);
    for (unsigned i = 0; i < frames && s->count < AUDIO_QUEUE; i++) {
        unsigned at = (s->read + s->count) % AUDIO_QUEUE;
        s->queue[at * 2] = in[i * 2];
        s->queue[at * 2 + 1] = in[i * 2 + 1];
        s->count++;
    }
    pthread_mutex_unlock(&s->lock);
    return true;
}

int SDL_GetAudioStreamQueued(SDL_AudioStream *s)
{
    if (!s) return -1;
    pthread_mutex_lock(&s->lock);
    int bytes = (int)s->count * 4;
    pthread_mutex_unlock(&s->lock);
    return bytes;
}

bool SDL_ClearAudioStream(SDL_AudioStream *s)
{
    if (!s) return false;
    pthread_mutex_lock(&s->lock);
    s->read = s->count = 0;
    pthread_mutex_unlock(&s->lock);
    return true;
}

bool SDL_SetAudioStreamGain(SDL_AudioStream *s, float gain) { if (!s) return false; s->gain = gain; return true; }
bool SDL_PauseAudioStreamDevice(SDL_AudioStream *s) { if (!s) return false; s->paused = 1; return true; }
bool SDL_ResumeAudioStreamDevice(SDL_AudioStream *s) { if (!s) return false; s->paused = 0; return true; }
SDL_AudioDeviceID SDL_GetAudioStreamDevice(SDL_AudioStream *s) { return s ? 1 : 0; }
bool SDL_GetAudioDeviceFormat(SDL_AudioDeviceID devid, SDL_AudioSpec *spec, int *frames)
{
    (void)devid;
    if (spec) { spec->format = SDL_AUDIO_S16; spec->channels = 2; spec->freq = AUDIO_RATE; }
    if (frames) *frames = AUDIO_PERIOD;
    return true;
}

void SDL_DestroyAudioStream(SDL_AudioStream *s)
{
    if (!s) return;
    s->running = 0;
    pthread_join(s->thread, NULL);
    sceAudioOutOutput(s->handle, NULL);
    sceAudioOutClose(s->handle);
    pthread_mutex_destroy(&s->lock);
    free(s);
}

/* ---------------- the DualShock 4: one PS4 gamepad, id 1 ---------------- */

struct SDL_Gamepad { int open; };
struct SDL_Joystick { int open; };
static SDL_Gamepad the_gamepad;
static SDL_Joystick the_joystick;
static int pad_handle = -1, pad_ready, pad_announced;
static OrbisPadData pad_state;
static Uint64 pad_read_at;
static pthread_mutex_t pad_lock = PTHREAD_MUTEX_INITIALIZER;

static void pad_open(void)
{
    pthread_mutex_lock(&pad_lock);
    if (!pad_ready) {
        sceUserServiceInitialize(NULL);
        if (scePadInit() >= 0) pad_ready = 1;
    }
    if (pad_ready && pad_handle < 0) {
        int user = -1;
        if (sceUserServiceGetInitialUser(&user) == 0 && user >= 0) {
            int h = scePadOpen(user, ORBIS_PAD_PORT_TYPE_STANDARD, 0, NULL);
            if (h >= 0) pad_handle = h;
        }
    }
    pthread_mutex_unlock(&pad_lock);
}

/* The pad's state, read again when the last read is older than 1 ms (the game asks per button). */
static const OrbisPadData *pad_read(void)
{
    pthread_mutex_lock(&pad_lock);
    Uint64 now = SDL_GetTicksNS();
    if (pad_handle >= 0 && now - pad_read_at > 1000000) {
        OrbisPadData d;
        memset(&d, 0, sizeof d);
        if (scePadReadState(pad_handle, &d) == 0) pad_state = d;
        pad_read_at = now;
    }
    pthread_mutex_unlock(&pad_lock);
    return &pad_state;
}

/* A pad that appears after the start (the player signs in later) reaches the game as SDL's
   joystick-added event, as on the other platforms. */
static void pad_events(void)
{
    if (!(initialized & (SDL_INIT_GAMEPAD | SDL_INIT_JOYSTICK)) || pad_announced) return;
    static Uint64 last_try;
    Uint64 now = SDL_GetTicks();
    if (pad_handle < 0 && now - last_try < 1000) return;
    last_try = now;
    if (pad_handle < 0) pad_open();
    if (pad_handle < 0) return;
    pad_announced = 1;
    SDL_Event e;
    memset(&e, 0, sizeof e);
    e.type = SDL_EVENT_JOYSTICK_ADDED;
    e.jdevice.which = 1;
    SDL_PushEvent(&e);
}

SDL_JoystickID *SDL_GetJoysticks(int *count)
{
    SDL_JoystickID *ids = calloc(2, sizeof *ids);
    int n = 0;
    if (ids && pad_handle >= 0) { ids[n++] = 1; pad_announced = 1; }
    if (count) *count = n;
    return ids;
}
bool SDL_IsGamepad(SDL_JoystickID id) { return id == 1 && pad_handle >= 0; }
SDL_Gamepad *SDL_OpenGamepad(SDL_JoystickID id)
{
    if (!SDL_IsGamepad(id)) { fail("no such gamepad"); return NULL; }
    the_gamepad.open = 1;
    return &the_gamepad;
}
SDL_Joystick *SDL_OpenJoystick(SDL_JoystickID id)
{
    if (!SDL_IsGamepad(id)) { fail("no such joystick"); return NULL; }
    the_joystick.open = 1;
    return &the_joystick;
}
void SDL_CloseGamepad(SDL_Gamepad *g) { if (g) g->open = 0; }
void SDL_CloseJoystick(SDL_Joystick *j) { if (j) j->open = 0; }
SDL_Joystick *SDL_GetGamepadJoystick(SDL_Gamepad *g) { (void)g; return &the_joystick; }
const char *SDL_GetGamepadName(SDL_Gamepad *g) { (void)g; return "DualShock 4"; }
const char *SDL_GetJoystickName(SDL_Joystick *j) { (void)j; return "DualShock 4"; }
SDL_GamepadType SDL_GetGamepadType(SDL_Gamepad *g) { (void)g; return SDL_GAMEPAD_TYPE_PS4; }
SDL_GamepadType SDL_GetRealGamepadType(SDL_Gamepad *g) { (void)g; return SDL_GAMEPAD_TYPE_PS4; }
SDL_JoystickType SDL_GetJoystickType(SDL_Joystick *j) { (void)j; return SDL_JOYSTICK_TYPE_GAMEPAD; }
char *SDL_GetGamepadMapping(SDL_Gamepad *g) { (void)g; return strdup("ps4:DualShock 4 (scePad)"); }
const char *SDL_GetGamepadStringForType(SDL_GamepadType type) { return type == SDL_GAMEPAD_TYPE_PS4 ? "ps4" : "unknown"; }
SDL_GamepadButtonLabel SDL_GetGamepadButtonLabel(SDL_Gamepad *g, SDL_GamepadButton b)
{
    (void)g;
    switch (b) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return SDL_GAMEPAD_BUTTON_LABEL_CROSS;
    case SDL_GAMEPAD_BUTTON_EAST: return SDL_GAMEPAD_BUTTON_LABEL_CIRCLE;
    case SDL_GAMEPAD_BUTTON_WEST: return SDL_GAMEPAD_BUTTON_LABEL_SQUARE;
    case SDL_GAMEPAD_BUTTON_NORTH: return SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE;
    default: return SDL_GAMEPAD_BUTTON_LABEL_UNKNOWN;
    }
}
SDL_PropertiesID SDL_GetGamepadProperties(SDL_Gamepad *g) { (void)g; return 0; }
SDL_PropertiesID SDL_GetJoystickProperties(SDL_Joystick *j) { (void)j; return 0; }
int SDL_GetNumJoystickButtons(SDL_Joystick *j) { (void)j; return 15; }
int SDL_GetNumJoystickAxes(SDL_Joystick *j) { (void)j; return 6; }
int SDL_GetNumJoystickHats(SDL_Joystick *j) { (void)j; return 0; }
Uint8 SDL_GetJoystickHat(SDL_Joystick *j, int hat) { (void)j; (void)hat; return SDL_HAT_CENTERED; }

/* The DS4's buttons by SDL's standard layout. BACK is the touchpad's click: the DS4's SHARE and PS
   buttons belong to the console. */
static unsigned button_bit(SDL_GamepadButton b)
{
    switch (b) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return ORBIS_PAD_BUTTON_CROSS;
    case SDL_GAMEPAD_BUTTON_EAST: return ORBIS_PAD_BUTTON_CIRCLE;
    case SDL_GAMEPAD_BUTTON_WEST: return ORBIS_PAD_BUTTON_SQUARE;
    case SDL_GAMEPAD_BUTTON_NORTH: return ORBIS_PAD_BUTTON_TRIANGLE;
    case SDL_GAMEPAD_BUTTON_BACK: return ORBIS_PAD_BUTTON_TOUCH_PAD;
    case SDL_GAMEPAD_BUTTON_START: return ORBIS_PAD_BUTTON_OPTIONS;
    case SDL_GAMEPAD_BUTTON_LEFT_STICK: return ORBIS_PAD_BUTTON_L3;
    case SDL_GAMEPAD_BUTTON_RIGHT_STICK: return ORBIS_PAD_BUTTON_R3;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return ORBIS_PAD_BUTTON_L1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return ORBIS_PAD_BUTTON_R1;
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return ORBIS_PAD_BUTTON_UP;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ORBIS_PAD_BUTTON_DOWN;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ORBIS_PAD_BUTTON_LEFT;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ORBIS_PAD_BUTTON_RIGHT;
    case SDL_GAMEPAD_BUTTON_TOUCHPAD: return ORBIS_PAD_BUTTON_TOUCH_PAD;
    default: return 0;
    }
}

bool SDL_GamepadHasButton(SDL_Gamepad *g, SDL_GamepadButton b) { (void)g; return button_bit(b) != 0; }
bool SDL_GamepadHasAxis(SDL_Gamepad *g, SDL_GamepadAxis a) { (void)g; return a >= 0 && a < SDL_GAMEPAD_AXIS_COUNT; }
bool SDL_GetGamepadButton(SDL_Gamepad *g, SDL_GamepadButton b)
{
    (void)g;
    unsigned bit = button_bit(b);
    return bit && (pad_read()->buttons & bit) && pad_state.connected;
}

static Sint16 stick_axis(uint8_t v)  /* 0..255, 128 centre -> SDL's -32768..32767 */
{
    int x = ((int)v - 128) * 256;
    return (Sint16)(x > 32767 ? 32767 : x);
}

Sint16 SDL_GetGamepadAxis(SDL_Gamepad *g, SDL_GamepadAxis a)
{
    (void)g;
    const OrbisPadData *d = pad_read();
    if (!d->connected) return 0;
    switch (a) {
    case SDL_GAMEPAD_AXIS_LEFTX: return stick_axis(d->leftStick.x);
    case SDL_GAMEPAD_AXIS_LEFTY: return stick_axis(d->leftStick.y);
    case SDL_GAMEPAD_AXIS_RIGHTX: return stick_axis(d->rightStick.x);
    case SDL_GAMEPAD_AXIS_RIGHTY: return stick_axis(d->rightStick.y);
    case SDL_GAMEPAD_AXIS_LEFT_TRIGGER: return (Sint16)(d->analogButtons.l2 * 32767 / 255);
    case SDL_GAMEPAD_AXIS_RIGHT_TRIGGER: return (Sint16)(d->analogButtons.r2 * 32767 / 255);
    default: return 0;
    }
}

bool SDL_GetJoystickButton(SDL_Joystick *j, int button) { (void)j; return SDL_GetGamepadButton(&the_gamepad, (SDL_GamepadButton)button); }
Sint16 SDL_GetJoystickAxis(SDL_Joystick *j, int axis) { (void)j; return SDL_GetGamepadAxis(&the_gamepad, (SDL_GamepadAxis)axis); }

bool SDL_RumbleGamepad(SDL_Gamepad *g, Uint16 low, Uint16 high, Uint32 ms)
{
    (void)g; (void)ms;
    if (pad_handle < 0) return fail("no pad");
    OrbisPadVibeParam vibe = {(uint8_t)(low >> 8), (uint8_t)(high >> 8)};
    return scePadSetVibration(pad_handle, &vibe) == 0;
}
bool SDL_RumbleJoystick(SDL_Joystick *j, Uint16 low, Uint16 high, Uint32 ms) { (void)j; return SDL_RumbleGamepad(&the_gamepad, low, high, ms); }
bool SDL_SetGamepadPlayerIndex(SDL_Gamepad *g, int index) { (void)g; (void)index; return true; }
int SDL_AddGamepadMappingsFromFile(const char *file) { (void)file; return 0; }

/* ---------------- files and paths ---------------- */

const char *SDL_GetBasePath(void)
{
    mkdir(PS4_BASE_PATH, 0777);
    return PS4_BASE_PATH;
}

bool SDL_GetPathInfo(const char *path, SDL_PathInfo *info)
{
    struct stat st;
    if (stat(path, &st)) {
        if (info) memset(info, 0, sizeof *info);
        return fail("no such path");
    }
    if (info) {
        info->type = S_ISDIR(st.st_mode) ? SDL_PATHTYPE_DIRECTORY : S_ISREG(st.st_mode) ? SDL_PATHTYPE_FILE : SDL_PATHTYPE_OTHER;
        info->size = (Uint64)st.st_size;
        info->create_time = info->modify_time = (SDL_Time)st.st_mtime * 1000000000;
        info->access_time = (SDL_Time)st.st_atime * 1000000000;
    }
    return true;
}

bool SDL_EnumerateDirectory(const char *path, SDL_EnumerateDirectoryCallback callback, void *userdata)
{
    DIR *dir = opendir(path);
    if (!dir) return fail("cannot open the folder");
    char dirname[1024];
    size_t n = strlen(path);
    snprintf(dirname, sizeof dirname, "%s%s", path, n && path[n - 1] == '/' ? "" : "/");
    bool ok = true;
    for (struct dirent *e; (e = readdir(dir));) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        SDL_EnumerationResult r = callback(userdata, dirname, e->d_name);
        if (r == SDL_ENUM_SUCCESS) break;
        if (r == SDL_ENUM_FAILURE) { ok = false; break; }
    }
    closedir(dir);
    return ok;
}

bool SDL_CreateDirectory(const char *path)
{
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(buf, 0777); *p = '/'; }
    return mkdir(buf, 0777) == 0 || errno == EEXIST;
}

/* ---------------- the rest ---------------- */

int SDL_GetSystemRAM(void) { return 8192; }
int SDL_GetNumLogicalCPUCores(void) { return 6; }  /* the cores a game gets on the PS4 */
void SDL_free(void *p) { free(p); }
void *SDL_realloc(void *p, size_t size) { return realloc(p, size); }
char *SDL_strdup(const char *s) { return strdup(s); }
int SDL_strcasecmp(const char *a, const char *b) { return strcasecmp(a, b); }
int SDL_toupper(int c) { return toupper(c); }
int SDL_isalnum(int c) { return isalnum(c); }
float SDL_floorf(float x) { return floorf(x); }

/* No image loading, child processes or Metal on the PS4: each reports failure, which the host
   handles (no window icon, no launcher hand-over). */
SDL_IOStream *SDL_IOFromConstMem(const void *mem, size_t size) { (void)mem; (void)size; fail("no IO streams"); return NULL; }
SDL_Surface *SDL_LoadPNG_IO(SDL_IOStream *src, bool closeio) { (void)src; (void)closeio; fail("no PNG loading"); return NULL; }
void SDL_DestroySurface(SDL_Surface *surface) { (void)surface; }
SDL_Process *SDL_CreateProcessWithProperties(SDL_PropertiesID props) { (void)props; fail("no processes"); return NULL; }
void SDL_DestroyProcess(SDL_Process *process) { (void)process; }
