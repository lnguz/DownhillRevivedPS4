/* PS4: <pthread.h> plus the GNU pthread_getname_np, which the console's libc lacks (the crash
   report names its thread with it); the name is not read back on the console. */
#ifndef DHD_PS4_PTHREAD_H
#define DHD_PS4_PTHREAD_H
#include_next <pthread.h>
#include <string.h>
static inline int pthread_getname_np(pthread_t thread, char *name, size_t size)
{
    (void)thread;
    if (size) { strncpy(name, "?", size); name[size - 1] = 0; }
    return 0;
}
#endif
