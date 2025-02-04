import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from tqdm import tqdm
from flask import url_for

# Set device (MPS for macOS, otherwise fallback to CPU/GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables for lazy loading
vl_chat_processor = None
vl_gpt = None

# Ensure generated image folder exists
os.makedirs("static/generated_images", exist_ok=True)

def load_model():
    """Load the model and processor only when needed."""
    global vl_chat_processor, vl_gpt

    if vl_chat_processor is None or vl_gpt is None:
        print("Loading processor and tokenizer...")
        model_path = "deepseek-ai/Janus-Pro-7B"
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        
        print("Loading model...")
        vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()
        print("Model loaded successfully!")

@torch.inference_mode()
def generate_image(prompt, temperature=0.8, parallel_size=1, cfg_weight=5, image_token_num_per_image=576, img_size=384, patch_size=16):
    """ Generate an AI-generated image based on user input and return the file path. """

    # Lazy-load model if not loaded
    load_model()

    print("Preparing prompt...")
    conversation = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt_text = sft_format + vl_chat_processor.image_start_tag

    print("Initializing generation process...")
    input_ids = vl_chat_processor.tokenizer.encode(prompt_text)
    input_ids = torch.LongTensor(input_ids).to(device)

    print("Setting up token generation...")
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    print("Beginning token generation...")
    outputs = None  # Store past_key_values
    for i in tqdm(range(image_token_num_per_image), desc="Generating image tokens"):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])

        # Classifier-Free Guidance (CFG)
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        # Sample next token
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # Prepare next token
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    print("Decoding generated tokens into image...")
    dec = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    # Save and return image path
    file_path = f"static/generated_images/img_{np.random.randint(10000)}.jpg"
    save_path = os.path.join(file_path)
    PIL.Image.fromarray(visual_img[0]).save(save_path)

    print(f"Image saved: {save_path}")
    return file_path  # Flask will serve this image