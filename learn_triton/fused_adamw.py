import torch 
import triton 
import triton.language as tl

@triton.jit 
def update_fn_kernel(p_ptr, grad_ptr, m_ptr, v_ptr, lr, beta1, beta2, eps, weight_decay, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets

    # load the slice of p 
    p = tl.load(offset_p_ptr, mask=mask)

    # load the slice of grad
    grad = tl.load(offset_grad_ptr, mask=mask)

    # gt <- grad 
    # mt <- beta1 * mt-1 + (1 - beta1) * gt
    # vt <- beta2 * vt-1 + (1 - beta2) * gt * gt
    # mthat <- mt / (1 - beta1^t)
    # vthat <- vt / (1 - beta2^t)
    # pt <- pt-1 - lr * mthat / (sqrt(vthat) + eps) - lr * weight_decay * pt-1

    m_prev = tl.load(m_ptr + offsets, mask=mask)
    v_prev = tl.load(v_ptr + offsets, mask=mask)
    mt = beta1 * m_prev + (1 - beta1) * grad
    vt = beta2 * v_prev + (1 - beta2) * grad * grad
    mthat = mt / (1 - beta1)
    vthat = vt / (1 - beta2)
    p = p - lr * mthat / (tl.sqrt(vthat) + eps) - lr * weight_decay * p

    # store the slice of p, m, v 
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(m_ptr + offsets, mt, mask=mask)
    tl.store(v_ptr + offsets, vt, mask=mask)

    # TODO are we losing precision for some reason? 


def fused_update_fn(p, grad, m, v, lr, beta1, beta2, eps, weight_decay):
    # find the current program
    # calculate the offsets based on the block size (what is block size in this case? its 1? no it should be a big chunk of the params, needs to be power of 2)

    # here's a mental example: 
    # pid = 1
    # block_size = 128 
    # block_start = pid * block_size = 128
    # offsets = block_start + tl.arange(0, block_size) = 128 + [0, 1, 2, 3, ... 127] = [128, 129, 130, ... 255]
    # mask = offsets < num_params = [128, 129, 130, ... 255] < 256 = [True, True, True, ... True]. say num params is like 254. last 2 are false
    n_elements = p.numel()

    BLOCK_SIZE=128
    grid = triton.cdiv(n_elements, BLOCK_SIZE)

    # note: we do not need to malloc the output here because we are just doing a simple update in place 
    # p, grad are passed as ptrs because they are tensors? 
    # lr, beta1, beta2, eps, weight_decay are passed as values because they are scalars?
    update_fn_kernel[(grid,)](p, grad, m, v, lr, beta1, beta2, eps, weight_decay, n_elements, BLOCK_SIZE=BLOCK_SIZE)

class FusedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FusedAdam, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                grad = p.grad 
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                state = self.state[p]

                if (len(state) == 0):
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                
                m = state['m']
                v = state['v']

                # TODO pass in momentum and anything else from state too here 
                fused_update_fn(
                    p, grad, m, v, lr, beta1, beta2, eps, weight_decay 
                )


if __name__ == "__main__":
    # initialize some random params and grads
    layer = torch.nn.Linear(256, 256, bias=False).cuda()
    params = list(layer.parameters())
    for p in params:
        p.grad = torch.randn_like(p)

    params_copy = []
    for p in params:
        clone = p.clone().detach().cuda()
        clone.requires_grad = True
        clone.grad = p.grad.clone().detach().cuda()
        params_copy.append(clone)
    
    # check that params and params copy are close
    for p, p_copy in zip(params, params_copy):
        assert torch.allclose(p, p_copy)
        assert torch.allclose(p.grad, p_copy.grad)
    

    adam = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    fused_adam = FusedAdam(params_copy, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # time 10000 steps 
    import time

    for optim in [adam, fused_adam]:
        start = time.time()
        for i in range(5):
            optim.step()
        end = time.time()
        print("time elapsed: ", end - start)
    print(p)
    print(p_copy)

    # check that results are the same 
    for p, p_copy in zip(params, params_copy):
        assert torch.allclose(p, p_copy)
        assert torch.allclose(p.grad, p_copy.grad)
