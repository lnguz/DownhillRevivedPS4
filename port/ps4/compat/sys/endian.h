/* PS4: FreeBSD's <sys/endian.h>, which SDL3's headers include on a FreeBSD target; the console's
   libc (musl-based) has the byte order macros in <endian.h>. */
#ifndef DHD_PS4_SYS_ENDIAN_H
#define DHD_PS4_SYS_ENDIAN_H
#include <endian.h>
#ifndef _BYTE_ORDER
#define _BYTE_ORDER BYTE_ORDER
#define _LITTLE_ENDIAN LITTLE_ENDIAN
#define _BIG_ENDIAN BIG_ENDIAN
#endif
#endif
