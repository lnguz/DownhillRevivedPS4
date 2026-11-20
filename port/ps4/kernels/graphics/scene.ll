; The graphics pipeline's second check (gnm.c): vertices from a buffer, a textured pixel stage
; whose body is scene.cl's (linked and inlined by gcn_kernels.py), for depth tests and blending.
; A vertex is two vec4s: position (screen x, y, z, 1) and (u, v, alpha, 1), the one parameter.

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)
declare float @llvm.amdgcn.interp.p1(float, i32 immarg, i32 immarg, i32)
declare float @llvm.amdgcn.interp.p2(float, float, i32 immarg, i32 immarg, i32)
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float)
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1)
declare <4 x float> @scene_shade(float, float, float, ptr addrspace(1))

define amdgpu_vs void @scene_vs(i64 inreg %vertices, i32 %vertex) {
  %base = inttoptr i64 %vertices to ptr addrspace(1)
  %i0 = mul i32 %vertex, 2
  %i1 = add i32 %i0, 1
  %p0 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 %i0
  %p1 = getelementptr <4 x float>, ptr addrspace(1) %base, i32 %i1
  %pos = load <4 x float>, ptr addrspace(1) %p0, align 16
  %par = load <4 x float>, ptr addrspace(1) %p1, align 16
  %x = extractelement <4 x float> %pos, i32 0
  %y = extractelement <4 x float> %pos, i32 1
  %z = extractelement <4 x float> %pos, i32 2
  %w = extractelement <4 x float> %pos, i32 3
  %u = extractelement <4 x float> %par, i32 0
  %v = extractelement <4 x float> %par, i32 1
  %a = extractelement <4 x float> %par, i32 2
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %u, float %v, float %a, float 1.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 false)
  ret void
}

define amdgpu_ps void @scene_ps(i64 inreg %texture, i32 inreg %prim_mask, <2 x float> %persp_sample, <2 x float> %persp_center) {
  %i = extractelement <2 x float> %persp_center, i32 0
  %j = extractelement <2 x float> %persp_center, i32 1
  %u1 = call float @llvm.amdgcn.interp.p1(float %i, i32 0, i32 0, i32 %prim_mask)
  %u = call float @llvm.amdgcn.interp.p2(float %u1, float %j, i32 0, i32 0, i32 %prim_mask)
  %v1 = call float @llvm.amdgcn.interp.p1(float %i, i32 1, i32 0, i32 %prim_mask)
  %v = call float @llvm.amdgcn.interp.p2(float %v1, float %j, i32 1, i32 0, i32 %prim_mask)
  %a1 = call float @llvm.amdgcn.interp.p1(float %i, i32 2, i32 0, i32 %prim_mask)
  %a = call float @llvm.amdgcn.interp.p2(float %a1, float %j, i32 2, i32 0, i32 %prim_mask)
  %t = inttoptr i64 %texture to ptr addrspace(1)
  %c = call <4 x float> @scene_shade(float %u, float %v, float %a, ptr addrspace(1) %t)
  %r = extractelement <4 x float> %c, i32 0
  %g = extractelement <4 x float> %c, i32 1
  %b = extractelement <4 x float> %c, i32 2
  %al = extractelement <4 x float> %c, i32 3
  ; FP16 export (SPI_SHADER_FP16_ABGR), packed two channels a register: the colour blender takes no
  ; 32-bit export into an 8-bit target (it wrote 0x020080ff for a blend through 32_ABGR)
  %rg = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %r, float %g)
  %ba = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %b, float %al)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %rg, <2 x half> %ba, i1 true, i1 true)
  ret void
}
