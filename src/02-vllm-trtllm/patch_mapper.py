import sys

p = "/root/thuan-engine-bench/patches/qwen3_5_weight_mapper.py"
s = open(p).read()

if "SM120 bench patch" in s:
    print("already patched")
    sys.exit(0)

# 1. _pack_projection_tensor: collapse scalar scales (identical replicas or
#    near-equal per-projection scales) into one scalar.
old_pack = """    def _pack_projection_tensor(self, tensors: list[torch.Tensor], num_groups: int) -> torch.Tensor:
        reference_shape = tensors[0].shape[1:]"""
new_pack = """    def _pack_projection_tensor(self, tensors: list[torch.Tensor], num_groups: int) -> torch.Tensor:
        # SM120 bench patch: ModelOpt NVFP4 checkpoints carry 0-dim per-tensor
        # scales (input_scale, weight_scale_2); a fused projection needs one
        # scalar, so take the max (exact when all are replicas of one scalar).
        if tensors[0].dim() == 0:
            vals = torch.stack([t.reshape(()).to(torch.float32) for t in tensors])
            if not torch.allclose(vals, vals.max().expand_as(vals)):
                print(f"[patch] WARNING: differing scalar scales {vals.tolist()}, using max")
            return vals.max().to(tensors[0].dtype)
        reference_shape = tensors[0].shape[1:]"""
assert old_pack in s, "pack anchor"
s = s.replace(old_pack, new_pack)

# 2. _split_qkv_tensor: replicate 0-dim scalars instead of splitting
old_split = """        expected_total = expected_q * 2 + expected_v
        assert tensor.shape[0] == expected_total, ("""
new_split = """        if tensor.dim() == 0:
            # SM120 bench patch: scalar per-tensor scale, same for q/k/v
            return tensor, tensor, tensor
        expected_total = expected_q * 2 + expected_v
        assert tensor.shape[0] == expected_total, ("""
assert old_split in s, "split anchor"
s = s.replace(old_split, new_split)

# 3. guard shape asserts at the call site
for name, row in [("q_tensor", "row_q"), ("k_tensor", "row_q"), ("v_tensor", "row_v")]:
    old = "assert %s.shape[0] == %s\n" % (name, row)
    new = "assert %s.dim() == 0 or %s.shape[0] == %s\n" % (name, name, row)
    assert old in s, name
    s = s.replace(old, new)
old_z = 'assert tensors["z"].shape[0] == row_v'
new_z = 'assert tensors["z"].dim() == 0 or tensors["z"].shape[0] == row_v'
assert old_z in s, "z anchor"
s = s.replace(old_z, new_z)

open(p, "w").write(s)
print("patch applied OK")
