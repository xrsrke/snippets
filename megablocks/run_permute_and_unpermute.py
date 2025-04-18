# NOTE: install from source, dont use the pip package because they doesnt have the permute function (not up to date)
# pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@main
# (env) (base) phuc_nguyen@ip-26-0-167-9:/fsx/phuc/temp/env_for_ref_moe/nanotron/megablocks$ python3 run_permute_and_unpermute.py
# indices (indices.shape=torch.Size([4, 2])): tensor([[1, 2],
#         [0, 1],
#         [0, 2],
#         [1, 2]], device='cuda:0', dtype=torch.int32)

# input_act (input_act.shape=torch.Size([4, 4])): tensor([[0., 0., 0., 0.],
#         [1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]], device='cuda:0')

# --------------------------------


# row_id_map (row_id_map.shape=torch.Size([8])): tensor([2, 0, 1, 4, 5, 3, 6, 7], device='cuda:0', dtype=torch.int32)

# permuted_inputs (permuted_inputs.shape=torch.Size([8, 4])): tensor([[1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [0., 0., 0., 0.],
#         [1., 1., 1., 1.],
#         [3., 3., 3., 3.],
#         [0., 0., 0., 0.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]], device='cuda:0')

# --------------------------------


# unpermute_outputs (unpermute_outputs.shape=torch.Size([4, 4])): tensor([[0., 0., 0., 0.],
#         [2., 2., 2., 2.],
#         [4., 4., 4., 4.],
#         [6., 6., 6., 6.]], device='cuda:0')


import torch
from grouped_gemm.ops import permute, unpermute

indices = torch.tensor([[1, 2], [0, 1], [0, 2], [1, 2]], dtype=torch.int32, device='cuda')
input_act = torch.tensor([[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3]], dtype=torch.float32, device='cuda')
probs = torch.ones_like(indices, dtype=torch.float32)
permuted_inputs, row_id_map = permute(input_act, indices)
unpermute_outputs = unpermute(permuted_inputs, row_id_map, probs)

print(f"indices (indices.shape={indices.shape}): {indices} \n")
print(f"input_act (input_act.shape={input_act.shape}): {input_act} \n")

print(f"-------------------------------- \n \n")
print(f"row_id_map (row_id_map.shape={row_id_map.shape}): {row_id_map} \n")
print(f"permuted_inputs (permuted_inputs.shape={permuted_inputs.shape}): {permuted_inputs} \n")

print(f"-------------------------------- \n \n")

print(f"unpermute_outputs (unpermute_outputs.shape={unpermute_outputs.shape}): {unpermute_outputs} \n")
