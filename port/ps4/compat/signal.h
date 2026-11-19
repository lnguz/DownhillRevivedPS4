/* PS4: OpenOrbis's <signal.h> names the union member __sa_sigaction but defines sa_sigaction as
   __sa_handler.sa_sigaction, so no code can set a SA_SIGINFO handler; this repairs the macro. */
#ifndef DHD_PS4_SIGNAL_H
#define DHD_PS4_SIGNAL_H
#include_next <signal.h>
#undef sa_sigaction
#define sa_sigaction __sa_handler.__sa_sigaction
#endif
