; The graphics pipeline's start-up check (gnm.c, gnm_triangle_check): one triangle in screen
; coordinates from the vertex index, one parameter (a dummy, the vertex stage must export one),
; and a pixel stage that writes red. LLVM IR for the amdgpu_vs / amdgpu_ps calling conventions
; (gcn_kernels.py builds it for amdgcn-mesa-mesa3d, GCN 1.1).

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)

define amdgpu_vs void @triangle_vs(i32 %vertex) {
  %is1 = icmp eq i32 %vertex, 1
  %is2 = icmp eq i32 %vertex, 2
  %x = select i1 %is1, float 200.0, float 20.0
  %y = select i1 %is2, float 200.0, float 20.0
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float 1.0, float 1.0, float 1.0, float 1.0, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float %x, float %y, float 0.0, float 1.0, i1 true, i1 false)
  ret void
}

define amdgpu_ps void @triangle_ps(i32 inreg %prim_mask, <2 x float> %persp_sample) {
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 1.0, float 0.0, float 0.0, float 1.0, i1 true, i1 true)
  ret void
}
