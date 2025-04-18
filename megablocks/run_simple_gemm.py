import torch
from grouped_gemm import ops
import numpy as np

def simple_reference_gmm(a, b, batch_sizes, trans_b=False):
    """Reference implementation of grouped matrix multiplication for clarity."""
    batch_sizes = batch_sizes.cpu().numpy()
    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)

# Simplified example with very small tensors and simple values
# Number of groups
z = 2  # Just two groups

# Dimensions
m = 2  # rows per group (smaller)
k = 2  # common dimension (smaller)
n = 2  # columns

# Create input A with simple integer values
# A is a (z*m, k) = (4, 2) tensor
a = torch.tensor([
    # First group (first 2 rows)
    [1, 2],  # row 0
    [3, 4],  # row 1
    
    # Second group (next 2 rows)
    [5, 6],  # row 2
    [7, 8]   # row 3
], device='cuda', dtype=torch.bfloat16)  # Using float32 instead of bfloat16 for clarity

# Create input B with simple integer values
# B is a (z, k, n) = (2, 2, 2) tensor
b = torch.tensor([
    # First B matrix
    [
        [1, 2],  # row 0
        [3, 4]   # row 1
    ],
    
    # Second B matrix
    [
        [5, 6],  # row 0
        [7, 8]   # row 1
    ]
], device='cuda', dtype=torch.bfloat16)

# Setup batch sizes - equal size for simplicity
batch_sizes = torch.tensor([m] * z, device='cuda')  # [2, 2]

print("Input tensor A (shape: {}):".format(a.shape))
print(a)
print("\nInput tensor B (shape: {}):".format(b.shape))
print(b)
print("\nBatch sizes:", batch_sizes)

# Run the grouped matrix multiplication
result = ops.gmm(a, b, batch_sizes.to("cpu"), trans_b=False)
reference = simple_reference_gmm(a, b, batch_sizes, trans_b=False)

print("\nResult from ops.gmm (shape: {}):".format(result.shape))
print(result)
print("\nReference result (shape: {}):".format(reference.shape))
print(reference)

print("\nExplanation of the operation with detailed math:")
start = 0
for i in range(z):
    size = batch_sizes[i].item()
    a_group = a[start:start+size]
    b_matrix = b[i]
    result_group = a_group @ b_matrix
    
    print(f"\nGroup {i+1}:")
    print(f"A[{start}:{start+size}] =")
    print(a_group)
    print(f"B[{i}] =")
    print(b_matrix)
    print("Matrix multiplication calculation:")
    
    # Show detailed calculations for this group
    for row_idx in range(a_group.shape[0]):
        for col_idx in range(b_matrix.shape[1]):
            a_row = a_group[row_idx]
            b_col = b_matrix[:, col_idx]
            
            calc = f"Row {row_idx}, Col {col_idx}: "
            calc += " + ".join([f"{a_row[k]} × {b_col[k]}" for k in range(k)])
            calc += f" = {a_row.dot(b_col)}"
            print(calc)
    
    print("Result =")
    print(result_group)
    
    start += size

print("\nSummary of grouped matrix multiplication:")
print("1. A is divided into groups according to batch_sizes")
print("2. Each group multiplies with its corresponding B matrix")
print("3. Results are concatenated to form the final output tensor")