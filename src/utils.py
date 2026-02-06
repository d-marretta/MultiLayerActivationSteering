import torch
from diffusers import StableDiffusionPipeline
import math
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from PIL import Image

def load_pipeline(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device="cuda"):
    """Load the Stable Diffusion pipeline with specified model and device."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)
    return pipe

def get_unet_layers(pipe, extract_resnet=False, extract_attentions=True):
    """Extract ResNet and/or Attention layers from the UNet of the Stable Diffusion pipeline."""

    assert extract_resnet or extract_attentions
    
    nets = {}
    
    for i, block in enumerate(pipe.unet.down_blocks):
        if extract_resnet:
            for j, resnet in enumerate(block.resnets):
               nets[f"down_block_{i}_resnet_{j}"] = resnet

        if hasattr(block, "attentions") and extract_attentions:
            for j, attn in enumerate(block.attentions):
                for k, transformer in enumerate(attn.transformer_blocks):
                    name = f"down_block_{i}_attn_{j}_trans_{k}_attn2" # Cross-attention
                    nets[name] = transformer.attn2
                    name = f"down_block_{i}_attn_{j}_trans_{k}_attn1" # Self-attention
                    nets[name] = transformer.attn1
                    name = f"down_block_{i}_attn_{j}_trans_{k}_ff"
                    nets[name] = transformer.ff

    if extract_resnet:
        for j, resnet in enumerate(pipe.unet.mid_block.resnets):
            nets[f"mid_block_resnet_{j}"] = resnet

    if hasattr(pipe.unet.mid_block, "attentions") and extract_attentions:
        for j, attn in enumerate(pipe.unet.mid_block.attentions):
            for k, transformer in enumerate(attn.transformer_blocks):
                name = f"mid_block_attn_{j}_trans_{k}_attn2" # Cross-attention
                nets[name] = transformer.attn2
                name = f"mid_block_attn_{j}_trans_{k}_attn1" # Self-attention
                nets[name] = transformer.attn1
                name = f"mid_block_attn_{j}_trans_{k}_ff"
                nets[name] = transformer.ff
                
    
    for i, block in enumerate(pipe.unet.up_blocks):
        if extract_resnet:
            for j, resnet in enumerate(block.resnets):
                nets[f"up_block_{i}_resnet_{j}"] = resnet

        if hasattr(block, "attentions") and extract_attentions:
            for j, attn in enumerate(block.attentions):
                for k, transformer in enumerate(attn.transformer_blocks):
                    name = f"up_block_{i}_attn_{j}_trans_{k}_attn2" # Cross-attention
                    nets[name] = transformer.attn2
                    name = f"up_block_{i}_attn_{j}_trans_{k}_attn1" # Self-attention
                    nets[name] = transformer.attn1
                    name = f"up_block_{i}_attn_{j}_trans_{k}_ff"
                    nets[name] = transformer.ff

    return nets


def show_images(images: list[Image.Image], prompts: list[str], cols: int = 2, width: int = 40) -> None:
    assert len(images) == len(prompts)

    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax in axes[len(images):]:
        ax.axis('off')

    for ax, img, prompt in zip(axes, images, prompts):
        ax.imshow(img)
        ax.axis('off')
        wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=width))
        ax.text(0.5, -0.05, wrapped_prompt, fontsize=10, ha='center', va='top', transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()