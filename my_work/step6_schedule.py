import torch
from step2_comms import PipelineComms
from step4_model import ShardedMLP

def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, batch, targets, hidden_dim, device):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.

    TODOs:
    - Receive input from previous stage if not first stage (requires_grad)
    - Forward batch through model
    - Send output to next stage if not last stage (detach)
    - Perform backward pass: 
        - If last stage, compute loss and call backward on it
        - Else, receive grad from next stage and call backward
    - Send grad to previous stage if not first stage
    - Return loss if last stage, else None
    """
    # TODO: If comms.rank == 0, use 'batch' directly; else, receive input
    # TODO: Forward pass through model
    # TODO: If not last stage, send output to next stage
    # TODO: Backward pass (different for last and non-last stage)
    # TODO: Send grad to previous stage if not first
    # TODO: Return loss if last stage, else None
    pass

def gpipe_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    GPipe Schedule: FWD all chunks -> BWD all chunks.
    """
    # TODO: Chunk the batches into microbatches and the targets in to microtargets
    # TODO: Initialize buffers for the input and activations
    # TODO: For i in [0..chunks):
    #     - If comms.rank == 0, use microbatch directly; else, receive input
    #     - Forward microbatch through model
    #     - If not last stage, send output to next stage
    #     - Append input/output to buffers
    # TODO: For i in [0..chunks):
    #     - Get inputs/outputs for this chunk from buffers
    #     - If last stage, compute loss and call backward
    #     - Else, receive grad from next stage and call backward
    #     - Send grad to previous stage if not first
    # TODO: Return loss if last stage, else None
    pass

def onef_oneb_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    1F1B Schedule: Interleaves Forward and Backward passes in a pipelined manner.
    """
    # TODO: Chunk the batches into microbatches and the targets in to microtargets
    # TODO: Initialize buffers for activations, gradients, etc.

    # Forward warmup: Fill the pipeline
    # for i in range(num_warmup_steps):
    #     - If comms.rank == 0, use microbatch directly; else, receive input
    #     - Forward microbatch through model
    #     - If not last stage, send output to next stage
    #     - Append input/output to buffers

    # 1F1B Steady State
    # for i in range(num_steady_steps):
    #     - Forward pass for new microbatch (as above)
    #     - Backward pass for previous microbatch
    #         - If last stage, compute loss and call backward
    #         - Else, receive grad from next stage and call backward
    #         - Send grad to previous stage if not first

    # Backward drain: Complete outstanding backward passes
    # for i in range(num_drain_steps):
    #     - Backward pass for remaining microbatches (as above)

    # TODO: Return loss if last stage, else None
        # 1. Prepare Data Slices
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)
    warmup = comms.world_size-comms.rank-1
    onefoneb = chunks - warmup
    
    # Storage for "Phase 2"
    input_buffers = [None] * chunks 
    output_buffers = [None] * chunks
    async_requests = []

    def forward(microbatchidx):
        if comms.rank == 0:
            input_data = micro_batches[microbatchidx]
        else:
            shape = (batch//chunks, hidden_dim)
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True

        # B. Forward Pass
        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[microbatchidx])
        else:
            output = model(input_data)
            req = comms.isend_forward(output.detach())
            async_requests.append(req)

        # D. Buffer for Backward
        input_buffers[microbatchidx] = input_data
        output_buffers[microbatchidx] = output # On last rank, this is the Loss
    
    def backward(microbatchidx):
        # Retrieve state from Phase 1
        input_data = input_buffers[microbatchidx]
        output = output_buffers[microbatchidx]
        
        if comms.rank == comms.world_size - 1:
            # On Last Rank, 'output' IS the loss
            loss = output / chunks
            loss.backward()
        else:
            # On other ranks, we need gradients from downstream
            grad_from_next = comms.recv_backward(output.shape, device)
            output.backward(grad_from_next)
            
        # Send gradients backward (if not first)
        if comms.rank != 0:
            comms.send_backward(input_data.grad)
        if comms.rank == comms.world_size - 1:
            return loss
    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(1, device=device)
    for i in range(warmup):
        forward(i)
    for i in range(onefoneb):
        forward(i+warmup)
        res = backward(i)
        if comms.rank == comms.world_size - 1:
            total_loss += res
    for i in range(warmup):
        backward(i+onefoneb)
    # Return loss across chunks (for logging) if last rank
    if comms.rank == comms.world_size - 1:
        return total_loss