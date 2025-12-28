from comms import PipelineComms
from model import ShardedMLP
from profiler import PipelineProfiler

def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, profiler: PipelineProfiler, batch, targets, hidden_dim, device):
    profiler.start_stage("step")
    
    # --- PHASE 1: FORWARD PASS ---
    with profiler.time_block("forward_get_input"):
        if comms.rank == 0:
            input_data = batch
        else:
            shape = (batch, hidden_dim)
            with profiler.time_block("forward_recv"):
                input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True
    
    with profiler.time_block("forward_compute"):
        output = model(input_data, targets if comms.rank == comms.world_size -1 else None)
    
    if not model.is_last:
        with profiler.time_block("forward_send"):
            comms.send_forward(output.detach())
    
    # --- PHASE 2: BACKWARD PASS ---
    with profiler.time_block("backward_get_grad"):
        if model.is_last:
            loss = output
            with profiler.time_block("backward_compute"):
                loss.backward()
            grad_to_send = input_data.grad 
        else:
            with profiler.time_block("backward_recv"):
                grad_from_next = comms.recv_backward(output.shape, device)
            with profiler.time_block("backward_compute"):
                output.backward(grad_from_next)
            grad_to_send = input_data.grad
    
    if not model.is_first:
        with profiler.time_block("backward_send"):
            comms.send_backward(grad_to_send)
    
    profiler.end_stage("step")
    return loss.item() if model.is_last else None