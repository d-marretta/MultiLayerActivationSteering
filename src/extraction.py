import torch
from diffusers import StableDiffusionPipeline


def collect_dataset_activations(
    pipe: StableDiffusionPipeline,
    forget_set: list[str],
    retain_set: list[str],
    total_steps: int,
    guidance: float,
    nets: dict,
    layers: list[str],
    timesteps: list[int]
):
    """Collect positive/negative activations for the specified prompts and layers."""
    forget_acts = []
    retain_acts = []

    for idx, (forget_prompt, retain_prompt) in enumerate(zip(forget_set, retain_set)):
        print(f'[{idx+1}] Extracting acts for forget prompt: {forget_prompt}')
        forget_act = get_average_activations(pipe, forget_prompt, total_steps, guidance, nets, layers, timesteps)

        print(f'[{idx+1}] Extracting acts for retain prompt: {retain_prompt}')
        retain_act = get_average_activations(pipe, retain_prompt, total_steps, guidance, nets, layers, timesteps)
        
        forget_acts.append(forget_act)
        retain_acts.append(retain_act)

    forget_layers = {}
    retain_layers = {}
    
    for l in layers:
        forget_layers[l] = torch.stack([f[l] for f in forget_acts], dim=0)
        retain_layers[l] = torch.stack([r[l] for r in retain_acts], dim=0)
        
    return forget_layers, retain_layers


def get_average_activations(
    pipe: StableDiffusionPipeline,
    prompt: str,
    total_steps: int,
    guidance: float,
    nets: dict,
    layers: list[str],
    timesteps: list[int]
):
    """Run the Stable Diffusion pipeline and collect average activations for the specified layers at given timesteps."""
    # designed to be simple, using batches would cause coherence issues when collecting acts.
    result = {}
    handles = []

    current_step = 0

    def save_act(name):
        def hook(module, input, output):           
            if current_step in timesteps:
                # UNet calculates noise prediction for both conditioned and unconditioned input, so we take the second
                residual = output[1] if isinstance(output, tuple) else output

                if residual[1].ndim == 3: # Channels x Width x Height
                    act = residual[1].mean(dim=(1, 2)).detach().cpu()
                elif residual[1].ndim == 2: # Tokens x Context 
                    act = residual[1].mean(dim=0).detach().cpu()
                else:
                    raise Exception(f'Unexpected activation shape {residual[1].shape} for {name}') 
                
                result.setdefault(name, []).append(act)
                
        return hook

    for l in layers:
        handles.append(
            nets[l].register_forward_hook(save_act(l))
        )

    def callback(pipeline, step_index, timestep, callback_kwargs):
        nonlocal current_step
        current_step = step_index

        return callback_kwargs
    
    try:
        images = pipe(
            prompt,
            num_inference_steps=total_steps,
            guidance_scale=guidance,
            callback_on_step_end=callback
        )
        
        return {
            layer: torch.stack(tensors, dim=0)
            for layer, tensors in result.items()
        } # [T, C, H, W]
    except Exception as e:
        raise e
    finally:
        for h in handles:
            h.remove()


def get_top_k_layers(results, k):
    """Extract the top-k layers for each timestep based on the computed layer_nav scores."""
    res = {}
    for timestep, top in results.items():
        res[timestep] = [x[0] for x in top[:k]]
    return res


def mask_vectors_by_top_k(steering_vectors, timesteps, top_k_per_timestep):
    """Zero out steering vectors for layers that are not in the top-k for each timestep."""
    masked_vectors = {l: v.clone() for l, v in steering_vectors.items()}
    
    for ts_index, step in enumerate(timesteps):
        active_layers = top_k_per_timestep.get(step, [])
        
        for layer_name, vector_tensor in masked_vectors.items():
            # If this layer is NOT in the active list for this step, zero it out
            if layer_name not in active_layers:
                vector_tensor[ts_index] = 0.0

    return masked_vectors