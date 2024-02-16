import os
import socket
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from utils import calculate_dimensions, bytes_to_nice_format

def recv_tensor(data):
    print("Received data")

WORKER_NAME = "RPC_GLOBAL_WORKER_{}"


def timed_send_recv(data, id, start_event, end_event):    
    start_event.record()
    for _ in range(1):
        rpc.rpc_sync(to=WORKER_NAME.format(1), func=recv_tensor, args=(data,))
    
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    size = data.numel() * 4 # 4 is fp32
    algbw = (size / duration) * 8 # 8 is bytes to bits
    n = dist.get_world_size()
    # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw reflects how optimally the hardware is used
    busbw = algbw * (2*(n - 1) / n)
    
    print(f"{id}:\n",
            f"duration: {duration:.3f} sec\n",
            f"algbw: {algbw/1e9:.3f} Gbps\n",
            f"busbw: {busbw / 1e9:.3f} Gbps"
    )



def run(local_rank):
    TRIALS = 2
    input_sizes = [(1, "MB"), (5, "MB"), (10, "MB"), (50, "MB"), (100, "MB"), (1, "GB"), (5, "GB"), (20, "GB")]

    hostname = socket.gethostname()
    id = f"{hostname}:{local_rank}"

    sizes = calculate_dimensions(input_sizes)
    
    for M, N in sizes:        
        # NOTE: these emulate the payload which will become a M * N * 4-sized tensor below
        data = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)
  
        data_size = bytes_to_nice_format(data.numel() * 4)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for trial in range(TRIALS):
            print(f"\n\n\n----------- [Trial {trial}][M={M}, N={N}, size={data_size} GB] ----------------")
            timed_send_recv(data, id, start_event, end_event)

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend)
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    options = rpc.TensorPipeRpcBackendOptions(
        _transports=["uv"],
    )
    
    if rank == 0:
        options.set_device_map(WORKER_NAME.format(1), {0: 1})
    
    rpc.init_rpc(
        name=WORKER_NAME.format(rank), rank=rank, world_size=world_size,
        rpc_backend_options=options
    )
    
    if rank == 0:
        fn(local_rank)
    
    dist.barrier()
    rpc.shutdown()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local_rank: %d" % local_rank)
    init_processes(local_rank=local_rank, fn=run)
