import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit(debug=True)
def adamw_kernel(
    params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    lr, beta1, beta2, eps, weight_decay,
    n_elements, step, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # mask = offsets < n_elements

    # Create block pointers
    params_block_ptr = tl.make_block_ptr(
        base=params_ptr, shape=(n_elements,), strides=(1,), offsets=(block_start,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    grads_block_ptr = tl.make_block_ptr(
        base=grads_ptr, shape=(n_elements,), strides=(1,), offsets=(block_start,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    exp_avg_block_ptr = tl.make_block_ptr(
        base=exp_avg_ptr, shape=(n_elements,), strides=(1,), offsets=(block_start,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    exp_avg_sq_block_ptr = tl.make_block_ptr(
        base=exp_avg_sq_ptr, shape=(n_elements,), strides=(1,), offsets=(block_start,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    params = tl.load(params_block_ptr)
    grads = tl.load(grads_block_ptr)
    exp_avg = tl.load(exp_avg_block_ptr)
    exp_avg_sq = tl.load(exp_avg_sq_block_ptr)

    tl.device_print("params", params)

    exp_avg = beta1 * exp_avg + (1 - beta1) * grads
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grads * grads

    bias_correction1 = 1 / (1 - libdevice.pow(beta1, step))
    bias_correction2 = 1 / (1 - libdevice.pow(beta2, step))
    exp_avg_hat = exp_avg * bias_correction1
    exp_avg_sq_hat = exp_avg_sq * bias_correction2

    normalized_grad = (exp_avg_hat / (tl.sqrt(exp_avg_sq_hat) + eps))
    weight_decay_grad = weight_decay * params
    params -= lr * (normalized_grad + weight_decay_grad)

    # Store updated values
    tl.store(params_block_ptr, params)
    tl.store(exp_avg_block_ptr, exp_avg)
    tl.store(exp_avg_sq_block_ptr, exp_avg_sq)

class TritonAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = 0

        self.exp_avg = [torch.zeros_like(p) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self._step += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            exp_avg = self.exp_avg[i]
            exp_avg_sq = self.exp_avg_sq[i]

            n_elements = p.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            adamw_kernel[grid](
                p, grad, exp_avg, exp_avg_sq,
                self.lr, self.beta1, self.beta2, self.eps, self.weight_decay,
                n_elements, self._step,
                BLOCK_SIZE=1024
            )

# Now let's compare it with PyTorch's AdamW

import torch.nn as nn

# Create a simple linear layer
input_dim, output_dim = 64, 32
x = torch.randn(16, input_dim, device="cuda")

# PyTorch implementation
torch_linear = nn.Linear(input_dim, output_dim).to("cuda")
torch_optimizer = torch.optim.AdamW(torch_linear.parameters(), lr=1e-3, weight_decay=0.01)

# Triton implementation
triton_linear = nn.Linear(input_dim, output_dim).to("cuda")
triton_optimizer = TritonAdamW(triton_linear.parameters(), lr=1e-3, weight_decay=0.01)

# Ensure both models have the same initial weights
triton_linear.load_state_dict(torch_linear.state_dict())

# Training loop
n_steps = 2
for step in range(n_steps):
    # PyTorch
    torch_optimizer.zero_grad()
    torch_output = torch_linear(x)
    torch_loss = torch_output.mean()
    torch_loss.backward()
    torch_optimizer.step()

    # Triton
    triton_optimizer.zero_grad()
    triton_output = triton_linear(x)
    triton_loss = triton_output.mean()
    triton_loss.backward()
    triton_optimizer.step()

# Compare final weights
torch_weights = torch_linear.weight.data
triton_weights = triton_linear.weight.data
weight_diff = torch.abs(torch_weights - triton_weights).mean().item()

print(f"Mean absolute difference in weights: {weight_diff:.6f}")
