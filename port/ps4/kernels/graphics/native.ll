; The native renderer's draw stages on the PS4 (port/ps4/render_ps4.c): entry points over
; native.cl's bodies (linked and inlined by gcn_kernels.py).
; Vertex: user data s2-s9 = the picture's vertices, its indices (an indexed picture), the draw's
; constants, the draw's first vertex or index, and whether the picture is indexed (USER_SGPR 10);
; it exports the position and two parameters, STQ+fog and the colour (0..255), interpolated
; without perspective (native.wgsl's Varyings).
; Pixel: user data s2-s3 = the draw's constants (USER_SGPR 4; PRIM_MASK in s4); the parameters at
; LINEAR_CENTER, the pixel's position; it discards (llvm.amdgcn.kill) or exports FP16.

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)
declare float @llvm.amdgcn.interp.p1(float, i32 immarg, i32 immarg, i32)
declare float @llvm.amdgcn.interp.p2(float, float, i32 immarg, i32 immarg, i32)
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float)
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1)
declare void @llvm.amdgcn.kill(i1)
declare void @native_vertex_body(i32, i32, i32, ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), ptr addrspace(5))
declare i32 @native_pixel_body(ptr addrspace(1), float, float, float, float, float, float, float, float, float, float, ptr addrspace(5))

define amdgpu_vs void @native_vs(i64 inreg %vertices, i64 inreg %indices, i64 inreg %constants, i32 inreg %first,
                                 i32 inreg %indexed, i32 %vertex) {
  %out = alloca [12 x float], align 4, addrspace(5)
  %vp = inttoptr i64 %vertices to ptr addrspace(1)
  %ip = inttoptr i64 %indices to ptr addrspace(1)
  %cp = inttoptr i64 %constants to ptr addrspace(1)
  call void @native_vertex_body(i32 %vertex, i32 %first, i32 %indexed, ptr addrspace(1) %vp, ptr addrspace(1) %ip,
                                ptr addrspace(1) %cp, ptr addrspace(5) %out)
  %o0p = getelementptr float, ptr addrspace(5) %out, i32 0
  %o1p = getelementptr float, ptr addrspace(5) %out, i32 1
  %o2p = getelementptr float, ptr addrspace(5) %out, i32 2
  %o3p = getelementptr float, ptr addrspace(5) %out, i32 3
  %o4p = getelementptr float, ptr addrspace(5) %out, i32 4
  %o5p = getelementptr float, ptr addrspace(5) %out, i32 5
  %o6p = getelementptr float, ptr addrspace(5) %out, i32 6
  %o7p = getelementptr float, ptr addrspace(5) %out, i32 7
  %o8p = getelementptr float, ptr addrspace(5) %out, i32 8
  %o9p = getelementptr float, ptr addrspace(5) %out, i32 9
  %o10p = getelementptr float, ptr addrspace(5) %out, i32 10
  %o11p = getelementptr float, ptr addrspace(5) %out, i32 11
  %o0 = load float, ptr addrspace(5) %o0p
  %o1 = load float, ptr addrspace(5) %o1p
  %o2 = load float, ptr addrspace(5) %o2p
  %o3 = load float, ptr addrspace(5) %o3p
  %o4 = load float, ptr addrspace(5) %o4p
  %o5 = load float, ptr addrspace(5) %o5p
  %o6 = load float, ptr addrspace(5) %o6p
  %o7 = load float, ptr addrspace(5) %o7p
  %o8 = load float, ptr addrspace(5) %o8p
  %o9 = load float, ptr addrspace(5) %o9p
  %o10 = load float, ptr addrspace(5) %o10p
  %o11 = load float, ptr addrspace(5) %o11p
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %o4, float %o5, float %o6, float %o7, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 33, i32 15, float %o8, float %o9, float %o10, float %o11, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float %o0, float %o1, float %o2, float %o3, i1 true, i1 false)
  ret void
}

define amdgpu_ps void @native_ps(i64 inreg %constants, i32 inreg %prim_mask, <2 x float> %persp_sample,
                                 <2 x float> %persp_center, <2 x float> %persp_centroid, <3 x float> %pull,
                                 <2 x float> %linear_sample, <2 x float> %linear_center, <2 x float> %linear_centroid,
                                 float %line_stipple, float %pos_x, float %pos_y) {
  %out = alloca [4 x float], align 4, addrspace(5)
  %i = extractelement <2 x float> %linear_center, i32 0
  %j = extractelement <2 x float> %linear_center, i32 1
  %s1 = call float @llvm.amdgcn.interp.p1(float %i, i32 0, i32 0, i32 %prim_mask)
  %s = call float @llvm.amdgcn.interp.p2(float %s1, float %j, i32 0, i32 0, i32 %prim_mask)
  %t1 = call float @llvm.amdgcn.interp.p1(float %i, i32 1, i32 0, i32 %prim_mask)
  %t = call float @llvm.amdgcn.interp.p2(float %t1, float %j, i32 1, i32 0, i32 %prim_mask)
  %q1 = call float @llvm.amdgcn.interp.p1(float %i, i32 2, i32 0, i32 %prim_mask)
  %q = call float @llvm.amdgcn.interp.p2(float %q1, float %j, i32 2, i32 0, i32 %prim_mask)
  %f1 = call float @llvm.amdgcn.interp.p1(float %i, i32 3, i32 0, i32 %prim_mask)
  %f = call float @llvm.amdgcn.interp.p2(float %f1, float %j, i32 3, i32 0, i32 %prim_mask)
  %r1 = call float @llvm.amdgcn.interp.p1(float %i, i32 0, i32 1, i32 %prim_mask)
  %r = call float @llvm.amdgcn.interp.p2(float %r1, float %j, i32 0, i32 1, i32 %prim_mask)
  %g1 = call float @llvm.amdgcn.interp.p1(float %i, i32 1, i32 1, i32 %prim_mask)
  %g = call float @llvm.amdgcn.interp.p2(float %g1, float %j, i32 1, i32 1, i32 %prim_mask)
  %b1 = call float @llvm.amdgcn.interp.p1(float %i, i32 2, i32 1, i32 %prim_mask)
  %b = call float @llvm.amdgcn.interp.p2(float %b1, float %j, i32 2, i32 1, i32 %prim_mask)
  %a1 = call float @llvm.amdgcn.interp.p1(float %i, i32 3, i32 1, i32 %prim_mask)
  %a = call float @llvm.amdgcn.interp.p2(float %a1, float %j, i32 3, i32 1, i32 %prim_mask)
  %cp = inttoptr i64 %constants to ptr addrspace(1)
  %keep = call i32 @native_pixel_body(ptr addrspace(1) %cp, float %s, float %t, float %q, float %f, float %r, float %g,
                                      float %b, float %a, float %pos_x, float %pos_y, ptr addrspace(5) %out)
  %live = icmp ne i32 %keep, 0
  call void @llvm.amdgcn.kill(i1 %live)
  %o0p = getelementptr float, ptr addrspace(5) %out, i32 0
  %o1p = getelementptr float, ptr addrspace(5) %out, i32 1
  %o2p = getelementptr float, ptr addrspace(5) %out, i32 2
  %o3p = getelementptr float, ptr addrspace(5) %out, i32 3
  %cr = load float, ptr addrspace(5) %o0p
  %cg = load float, ptr addrspace(5) %o1p
  %cb = load float, ptr addrspace(5) %o2p
  %ca = load float, ptr addrspace(5) %o3p
  %rg = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %cr, float %cg)
  %ba = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %cb, float %ca)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %rg, <2 x half> %ba, i1 true, i1 true)
  ret void
}
