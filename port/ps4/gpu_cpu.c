/* The GS renderer on the PS4 (port/ps4/PLAN.md, phase 4): the exact front end gpu_exact.inc over a
   CPU device, raster.wgsl's work done by the console's cores. The PS4 has neither Vulkan nor Metal
   nor Direct3D, so the Rust/wgpu renderer (port/gs/gpu_rust.c) does not build there.

   Phase 3 (the host) runs the game without a picture: this first version reports no device, as
   gpu_rust.c does on a machine without a GPU, and the game carries on headless. */
#include <stdint.h>
#include <string.h>

#include "../gs/gs.h"
#include "../gs/gpu.h"


static unsigned render_absence;

int gs_gpu_init(Gs *g) { (void)g; return 0; }
void gs_gpu_destroy(Gs *g) { (void)g; }
void gs_gpu_profile(Gs *g, double seconds) { (void)g; (void)seconds; }
int gs_gpu_display(Gs *g, unsigned *width, unsigned *height) { (void)g; *width = *height = 0; return 0; }
int gs_gpu_capture(Gs *g, uint32_t *pixels) { (void)g; (void)pixels; return 0; }
const char *gs_gpu_backend_name(const Gs *g) { (void)g; return "none"; }
int gs_gpu_surface(Gs *g, const GsWindow *window) { (void)g; (void)window; return 0; }
int gs_gpu_surface_reset(Gs *g, const GsWindow *window) { (void)g; (void)window; return 0; }
unsigned gs_render_leave(void) { return ++render_absence; }
unsigned gs_render_generation(void) { return render_absence; }
unsigned gs_render_current(void) { return render_absence; }
int gs_render_release(unsigned generation) { return generation == render_absence; }
int gs_render_quiet(void) { return 1; }
int gs_gpu_present(Gs *g, const uint32_t *pixels, unsigned width, unsigned height,
                   unsigned window_width, unsigned window_height)
{
    (void)g; (void)pixels; (void)width; (void)height; (void)window_width; (void)window_height;
    return 0;
}
int gs_gpu_lost(void) { return 0; }
int gs_gpu_native(const Gs *g) { (void)g; return 0; }
void gs_native_vsync(Gs *g, unsigned *width, unsigned *height, int resize)
{
    (void)g; (void)resize; *width = *height = 0;
}
void gs_native_vsync_handed(Gs *g, unsigned *width, unsigned *height, int resize, double due, double began,
                            int flat_limit, int card_2d, double output_aspect)
{
    (void)g; (void)resize; (void)due; (void)began; (void)flat_limit; (void)card_2d; (void)output_aspect;
    *width = *height = 0;
}
void gs_native_bench(Gs *g, int end) { (void)g; (void)end; }
void gs_native_lock(Gs *g, int lock) { (void)g; (void)lock; }
void gs_native_present_options(Gs *g) { (void)g; }
uint32_t *gs_native_picture(Gs *g, unsigned *width, unsigned *height) { (void)g; *width = *height = 0; return NULL; }
void gs_native_movie(Gs *g, const uint32_t *rgba, unsigned width, unsigned height, unsigned rows)
{
    (void)g; (void)rgba; (void)width; (void)height; (void)rows;
}

/* gpu_rust.c's and native_core.inc's other calls */
void gs_gpu_counter_apply(void) {}
int gs_native_card(void) { return 0; }  /* the native renderer's card mode: none */
/* The renderer's scale (1x on the CPU device); 2 asks whether a software adapter draws: none. */
int host_service_gs_scale(int asked) { return asked == 2 ? 0 : 1; }
