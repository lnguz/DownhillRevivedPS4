/* PS4: OpenOrbis's <sys/ioctl.h> keeps only Linux's names (LINUX_FIONREAD); the console's FreeBSD
   kernel takes FreeBSD's request numbers. */
#ifndef DHD_PS4_SYS_IOCTL_H
#define DHD_PS4_SYS_IOCTL_H
#include_next <sys/ioctl.h>
#ifndef FIONREAD
#define FIONREAD 0x4004667fUL  /* FreeBSD: _IOR('f', 127, int) */
#endif
#endif
