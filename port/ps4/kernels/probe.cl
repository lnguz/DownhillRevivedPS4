/* gnm.c's start-up check of the GPU path: each of count words becomes word * 3 + add. */
kernel void probe(global uint *data, uint count, uint add)
{
    uint i = __builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x();
    if (i < count) data[i] = data[i] * 3u + add;
}
