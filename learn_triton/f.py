import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Define the Triton kernel for the AdamW optimizer
@triton.jit
def adamw_kernel(
    param_ptr,   # Pointer to parameters
    grad_ptr,    # Pointer to gradients
    m_ptr,       # Pointer to first moment estimates
    v_ptr,       # Pointer to second moment estimates
    lr,          # Learning rate
    beta1,       # Exponential decay rate for the first moment estimates
    beta2,       # Exponential decay rate for the second moment estimates
    eps,         # Term added to the denominator to improve numerical stability
    weight_decay,# Weight decay (L2 penalty)
    step,        # Training step
    n_elements,  # Number of elements to process
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Create block pointers using make_block_ptr
    param_ptrs = tl.make_block_ptr(
        base=param_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(offsets,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    grad_ptrs = tl.make_block_ptr(
        base=grad_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(offsets,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    m_ptrs = tl.make_block_ptr(
        base=m_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(offsets,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(offsets,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )

    # Load parameters and gradients
    params = tl.load(param_ptrs, mask=mask)
    grads = tl.load(grad_ptrs, mask=mask)
    m = tl.load(m_ptrs, mask=mask)
    v = tl.load(v_ptrs, mask=mask)

    # Apply weight decay
    params = params - lr * weight_decay * params

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grads
    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * grads * grads

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1 ** step)
    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2 ** step)

    # Update parameters
    params = params - lr * m_hat / (tl.sqrt(v_hat) + eps)

    # Store updated parameters and states
    tl.store(param_ptrs, params, mask=mask)
    tl.store(m_ptrs, m, mask=mask)
    tl.store(v_ptrs, v, mask=mask)

# Simple neural network model with nn.Linear
class SimpleNet(nn.Module):
    def __init__(self, input_size=128, output_size=64):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Function to train the model using the custom Triton AdamW optimizer
def train_with_triton(model, criterion, input_data, target_data, num_epochs=10):
    # Initialize optimizer states
    state = {}
    for param in model.parameters():
        state[param] = {
            'm': torch.zeros_like(param.data),
            'v': torch.zeros_like(param.data),
            'step': 0
        }

    # Hyperparameters
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    BLOCK_SIZE = 1024

    loss_history = []

    for epoch in range(num_epochs):
        # Zero gradients
        model.zero_grad()

        # Forward pass
        outputs = model(input_data)
        loss = criterion(outputs, target_data)
        loss_history.append(loss.item())

        # Backward pass
        loss.backward()

        # Update parameters using Triton AdamW optimizer
        for param in model.parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            param_data = param.data
            m = state[param]['m']
            v = state[param]['v']
            state[param]['step'] += 1
            step = state[param]['step']

            n_elements = param_data.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

            # Launch Triton kernel
            #param_ptr,   # Pointer to parameters
            # grad_ptr,    # Pointer to gradients
            # m_ptr,       # Pointer to first moment estimates
            # v_ptr,       # Pointer to second moment estimates
            # lr,          # Learning rate
            # beta1,       # Exponential decay rate for the first moment estimates
            # beta2,       # Exponential decay rate for the second moment estimates
            # eps,         # Term added to the denominator to improve numerical stability
            # weight_decay,# Weight decay (L2 penalty)
            # step,        # Training step
            # n_elements,  # Number of elements to process
            # BLOCK_SIZE: tl.constexpr,
            adamw_kernel[grid](
                param_data, grad, m, v,
                lr, beta1, beta2, eps, weight_decay, step, n_elements,
                BLOCK_SIZE
            )

            assert 1 == 1

    return loss_history

# Function to train the model using PyTorch's built-in AdamW optimizer
def train_with_torch(model, criterion, input_data, target_data, num_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_history = []

    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_data)
        loss = criterion(outputs, target_data)
        loss_history.append(loss.item())

        # Backward pass and update
        loss.backward()
        optimizer.step()

    return loss_history

# Main function to compare both optimizers
def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize models
    model_triton = SimpleNet().to(device)
    model_torch = SimpleNet().to(device)

    # Use the same initial weights for fair comparison
    model_torch.load_state_dict(model_triton.state_dict())

    # Criterion and data
    criterion = nn.MSELoss()
    input_data = torch.randn(32, 128).to(device)
    target_data = torch.randn(32, 64).to(device)

    # Train with Triton AdamW
    loss_triton = train_with_triton(model_triton, criterion, input_data, target_data)

    # Train with PyTorch AdamW
    loss_torch = train_with_torch(model_torch, criterion, input_data, target_data)

    # Print loss history for comparison
    print("Loss history with Triton AdamW:")
    print(loss_triton)
    print("\nLoss history with PyTorch AdamW:")
    print(loss_torch)

    # Compare final parameters
    param_difference = 0.0
    for p1, p2 in zip(model_triton.parameters(), model_torch.parameters()):
        param_difference += torch.norm(p1.data - p2.data).item()

    print("\nTotal parameter difference between Triton and PyTorch models:")
    print(param_difference)

if __name__ == '__main__':
    main()