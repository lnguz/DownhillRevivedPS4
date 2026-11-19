# Downhill Revived for PS4: plan

A native PlayStation 4 port (jailbroken console, GoldHEN, OpenOrbis toolchain) of the game core
this repository builds. Bring your own disc: the package carries no game data; the game reads
`SCUS_971.77` and the disc's files from your own image, as on every other platform.

## How the game is built here

1. `tools/split.py` splits your `SCUS_971.77` into `asm/` and `assets/` (game-derived, gitignored).
2. `port/tools/build_wasm.py` builds the portable game core, `build/wasm/dhd_core.wasm`
   (3815 game units, the recompiled assembly, the module-side runtime and HLE). It imports 148
   `host_api` functions (`port/host/host_api.h`): that list is what a platform provides.
3. Each platform turns the module into C with `wasm2c` and links it with the native host
   (`port/host64`, `port/host`, `port/hle`, `port/gs`) and a renderer.

The PS4 needs no change to steps 1 and 2. Step 3 needs a PS4 host.

## What the PS4 host needs

| Part | Other platforms | PS4 |
|---|---|---|
| Module C | wasm2c, guard-page bounds checks | wasm2c with explicit bounds checks (`WASM_RT_MEMCHECK_BOUNDS_CHECK`), as iOS (finding 224) |
| Guest memory | 8 GiB reservation, shared-memory aliases, commit on fault (SIGSEGV) | the iOS bounds-check layout; memory committed up front (the PS2 had 32 MiB) |
| EE fibers | `fiber_sysv64.c` | the same (x86-64 System V; the PS4 is FreeBSD-based) |
| Window, sound, controllers, threads, timers, files | SDL3 (about 160 functions used) | a small SDL3 compatibility layer over `sceVideoOut`, `sceAudioOut`, `scePad`, pthreads and libkernel |
| GS renderer | `gpu_exact.inc` (CPU front end) + the Rust/wgpu renderer (`raster.wgsl` compute shaders on D3D12, Vulkan or Metal) | the same front end over a **CPU device**: the `op_*` seam of `gpu_exact.inc` implemented in C, `raster.wgsl` translated to multithreaded C (the PS4 has no Vulkan) |
| Launcher, online, updates | Rust / web launcher | not ported at first: the game starts straight from the disc image |
| Package | installers, APK, IPA | `.pkg` (title ID, icon, `/data/...` for saves and the disc image) |

## Phases

1. Core: build `dhd_core.wasm` on Linux/WSL from your disc. **Done** (228 s, 3815/3815 units).
2. Module C for PS4: wasm2c output compiles with the OpenOrbis toolchain (`-march=btver2`,
   Jaguar). **Done**: the 8 wasm2c parts compile with LLVM 23 for `x86_64-pc-freebsd12-elf` with
   the bounds-check defines, no errors, 79 s on 8 threads (about 37 MB of objects).
3. Host: the SDL3 layer, guest memory, fibers, sound and controllers; the game runs headless
   (no picture) and logs its progress.
4. Renderer: the CPU device behind `gpu_exact.inc`; first picture, then speed (tiles across the
   Jaguar cores, 1x resolution).
5. Package and test on the console; measure frame rate.

## Risks

- The CPU renderer's speed on the Jaguar cores (1.6 GHz). 30 fps at 1x is the first target.
- Memory: the module and the host must fit what a homebrew app gets.
