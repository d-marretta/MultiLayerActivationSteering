import torch
from diffusers import StableDiffusionPipeline

def apply_steering(x, r, lam=-1.0):
    """Apply steering to the activations x using the steering vector r and strength lam."""
    if torch.all(r == 0).item():
        return x

    r = r.to(x.device, x.dtype)
    r /= r.norm()
        
    if x.ndim == 4:  # [B, C, H, W]
        r = r[None, :, None, None] # shape [B, C, 1, 1]
        channel_dim = 1
    elif x.ndim == 3: # [B, T, C] (ff layers)
        r = r[None, None, :] # shape [B, C]
        channel_dim = 2
        
    dot_product = (x * r).sum(dim=channel_dim, keepdim=True)
    
    return x + (lam * dot_product * r)


def apply_steering_pca(x, r, pca_data: tuple[torch.Tensor, torch.Tensor], lam: float = -1.0):
    """Apply steering to the activations x using the steering vector r and strength lam, with PCA-based dimensionality reduction."""
    if torch.all(r == 0).item():
        return x

    r = r.to(x.device, x.dtype)

    pcs = pca_data[0].half().to(x.device)
    mean = pca_data[1].half().to(x.device)

    

    x_compressed = (x - mean) @ pcs
    x_compressed = x @ pcs
    r_compressed = r @ pcs
    r_constructed = (r_compressed @ pcs.T) + mean
    
    r_compressed /= r_compressed.norm()
    r_constructed /= r_constructed.norm()
        
    if x.ndim == 4:  # [B, C, H, W]
        r = r[None, :, None, None] # shape [B, C, 1, 1]
        channel_dim = 1
    elif x.ndim == 3: # [B, T, C] (attention layers)
        r = r[None, None, :] # shape [B, C]
        channel_dim = 2
        
    dot_product = (x_compressed * r_compressed).sum(dim=channel_dim, keepdim=True)
    
    return x + (lam * dot_product * r_constructed)


def generate_with_steering(
    pipe: StableDiffusionPipeline,
    prompt: str,
    guidance: float,
    nets: dict,
    steering_vectors: dict[str, torch.Tensor],
    timesteps: list[int],
    inference_steps: int,
    lam: float,
    pca_dict: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]
):
    """Generate images using the Stable Diffusion pipeline with steering applied to specified layers at given timesteps."""
    handles = []

    current_step = 0

    def steering_hook(layer: str, steering_vector: torch.Tensor):
        ts_index = 0
        
        def hook(module, inp, out):
            nonlocal ts_index
            

            # out can be tensor or (hidden, tensor)
            if isinstance(out, tuple):
                hidden, activation = out
            else:
                hidden, activation = None, out  # activation: [B, C, H, W]

            if current_step in timesteps: 
                
                B = activation.size(0)

                assert B % 2 == 0
                
                x = activation[B // 2 :, :, :]

                pca_data = pca_dict.get(layer, {}).get(current_step, None) if pca_dict is not None else None

                if pca_data is not None:
                    x_steered = apply_steering_pca(x, steering_vector[ts_index], pca_data, lam)
                else:
                    x_steered = apply_steering(x, steering_vector[ts_index], lam)
                
                activation[B // 2 :] = x_steered
                    
                ts_index += 1

            if hidden is None:
                return activation
            else:
                return (hidden, activation)

        return hook

    for layer, steering_vector in steering_vectors.items():
        handles.append(
            nets[layer].register_forward_hook(steering_hook(layer, steering_vector))
        )

    def callback(pipeline, step_index, timestep, callback_kwargs):
        nonlocal current_step
        current_step = step_index

        return callback_kwargs
    
    try:
        return pipe(
            prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance,
            callback_on_step_end=callback,
            generator=torch.Generator(device="cuda").manual_seed(362)
        ).images
    except Exception as e:
        raise e
    finally:
        for h in handles:
            h.remove()

