import torch
from comms import init_distributed, PipelineComms

def ping_pong():
    rank, world_size, device = init_distributed()
    # play with the barrier!
    torch.distributed.barrier()
    print(rank, world_size, device)
    comms = PipelineComms(rank, world_size)

    if rank == 0:
        tensor = torch.rand(3).to(device)
        print(f"Rank 0: Sending {tensor}")
        comms.send_forward(tensor)
    elif rank == 1:
        # Must know shape in advance!
        shape = (3,)
        received = comms.recv_forward(shape, device)
        print(f"Rank 1: Received {received}")

if __name__ == "__main__":
    ping_pong()