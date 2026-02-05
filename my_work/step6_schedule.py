import torch
from step2_comms import PipelineComms
from step4_model import ShardedMLP

CHUNKS = 4


def naive_pipeline_step(
    model: ShardedMLP, comms: PipelineComms, batch, targets, hidden_dim, device
):
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
    if comms.rank == 0:
        input_data = batch
    else:
        shape = (batch, hidden_dim)
        comms.recv_forward(shape, device)
    # TODO: Forward pass through model
    output = model(input_data, targets)
    if model.rank != model.world_size - 1:
        comms.send_forward(output.detach())
    # TODO: If not last stage, send output to next stage
    if model.rank == model.world_size - 1:
        loss = output
        loss.backward()
    # TODO: Backward pass (different for last and non-last stage)
    else:
        gradients = comms.recv_backward(output.shape, device)
        output.backward(gradients)

    if model.rank != 0:
        comms.send_backward(input_data.grad)
    # TODO: Send grad to previous stage if not first
    # TODO: Return loss if last stage, else None
    if model.rank == model.world_size - 1:
        return loss
    # pass


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
    if model.rank == 0:
        micro_batches = torch.chunk(batch / chunks, chunks)
    if model.rank == model.world_size - 1:
        micro_targets = torch.chunk(targets, chunks)

    # manual activations are stored here
    # backwards will be done here
    input_buffer = []
    output_buffer = []

    for i in range(chunks):
        if comms.rank == 0:
            input_data = micro_batches[i]
        else:
            shape = (micro_batches[i], hidden_dim)
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True
        if model.rank == model.world_size - 1:
            output = model(input_data, micro_targets[i])
        else:
            output = model(input_data)
        if model.rank != model.world_size - 1:
            comms.send_forward(output.detach())
        input_buffer.append(input_data)
        output_buffer.append(output)
    total_loss = torch.zeros(output.shape)
    for i in range(chunks):
        input_data = input_buffer[i]
        output = output_buffer[i]

        if model.rank == model.world_size - 1:
            loss = output
            loss.backward()
            total_loss += loss
        else:
            gradients = comms.recv_backward(output.shape, device)
            output.backward(gradients)
        if model.rank != 0:
            comms.send_backward(input_data.grad)
    if model.rank == model.world_size - 1:
        return loss

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
    pass
