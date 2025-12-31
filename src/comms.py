import torch
import torch.distributed as dist
import os

def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # 1. Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 2. Set Device
    if torch.cuda.is_available():
        # each conditional statement returns the device type
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        # >>> torch.device("mps")
        # device(type='mps')
        # device = torch.device("mps")
        # mps doesn't work :(
        device = torch.device("cpu")
    elif torch.cpu.is_available():
        device = torch.device("cpu")
    else:
        exit()
    # 3. Initialize Group
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:        
        # The code dist.init_process_group(...) is a Global State Setter.
        #  It initializes the background communication threads (C++ NCCL backend).
        # It sets up the "phone lines" so Process 0 can send data to Process 1.
        # Once called, this state persists until the program ends or you call destroy_process_group().
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    return rank, world_size, device

class PipelineComms:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # Define Neighbors
        # If I am Rank 0, I have no previous neighbor (None)
        self.prev_rank = rank - 1 if rank > 0 else None
        # If I am the last Rank, I have no next neighbor (None)
        self.next_rank = rank + 1 if rank < world_size - 1 else None

    def send_forward(self, tensor):
        """Send activation to the next GPU."""
        # .contiguous() is required before sending
        print(f"[Rank {self.rank}] send_forward() CALLED - BLOCKING until Rank {self.next_rank} calls recv_forward()")
        dist.send(tensor.contiguous(), dst=self.next_rank)
        print(f"[Rank {self.rank}] send_forward() COMPLETE - unblocked")

    def recv_forward(self, shape, device, dtype=torch.float32):
        """Receive activation from the previous GPU."""
        # We must allocate an empty buffer to receive the data
        print(f"[Rank {self.rank}] recv_forward() CALLED - BLOCKING until Rank {self.prev_rank} calls send_forward()")
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.prev_rank)
        print(f"[Rank {self.rank}] recv_forward() COMPLETE - unblocked")
        return tensor

    def send_backward(self, tensor):
        """Send gradients back to the previous GPU."""
        # Blocking communication (dist.send) means 
        # the program waits until the send is complete 
        # before proceeding, which is simple and easier
        # to reason about. Async (isend) allows overlapping
        # computation and communication, 
        # increasing efficiency and complexity.
        print(f"[Rank {self.rank}] send_backward() CALLED - BLOCKING until Rank {self.prev_rank} calls recv_backward()")
        dist.send(tensor.contiguous(), dst=self.prev_rank)
        print(f"[Rank {self.rank}] send_backward() COMPLETE - unblocked")

    def recv_backward(self, shape, device, dtype=torch.float32):
        """Receive gradients from the next GPU."""
        print(f"[Rank {self.rank}] recv_backward() CALLED - BLOCKING until Rank {self.next_rank} calls send_backward()")
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.next_rank)
        print(f"[Rank {self.rank}] recv_backward() COMPLETE - unblocked")
        return tensor
    
    def isend_forward(self, tensor):
        print(f"[Rank {self.rank}] isend_forward() CALLED - ASYNC (returns immediately)")
        return dist.isend(tensor.contiguous(), dst=self.next_rank)