# import torch
# from grouped_gemm.ops import permute, unpermute

# permuted_inputs, row_id_map = permute(inputs, indices)

# print(f"indices (indices.shape={indices.shape}): {indices} \n")
# print(f"permuted_inputs (permuted_inputs.shape={permuted_inputs.shape}): {permuted_inputs} \n")


import torch
from grouped_gemm.ops import permute, unpermute

indices = torch.tensor([[2],
        [3],
        [1],
        [3]], device='cuda', dtype=torch.int32)
# indices = torch.tensor([[1, 2], [0, 1], [0, 2], [1, 2]], dtype=torch.int32, device='cuda')

# input_act = torch.tensor([[0., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 1.],
#         [2., 2., 2., 2., 2., 2.],
#         [3., 3., 3., 3., 3., 3.]], device='cuda', dtype=torch.float32)

# input_act = torch.tensor([[0.,0.,0.,0.], [1.,1.,1.,1.], [2.,2.,2.,2.], [3.,3.,3.,3.]], dtype=torch.float32, device='cuda')
input_act = torch.tensor([[0.,0.,0.,0.], [1.,1.,1.,1.], [2.,2.,2.,2.], [3.,3.,3.,3.]], dtype=torch.float32, device='cuda')

permuted_inputs, row_id_map = permute(input_act, indices)

print(f"indices (indices.shape={indices.shape}): {indices} \n")
print(f"input_act (input_act.shape={input_act.shape}): {input_act} \n")

print(f"-------------------------------- \n \n")
# print(f"row_id_map (row_id_map.shape={row_id_map.shape}): {row_id_map} \n")
print(f"permuted_inputs (permuted_inputs.shape={permuted_inputs.shape}): {permuted_inputs} \n")
