"""
This script runs inference on the deepseek-coder-1.3b-instruct llm, generating 200 tokens for each of the prompts
contained in prompts.txt.

Inference is also run on the LSAE for each layer on each generated token, and the activation strength for all 24*4096 
features are recorded, totaling 24M recorded feature activations per prompt.

Feature activation strengths are saved to npy file "all_features_out.npy", generated tokens are saved to all_tokens_out.csv, 
and all model inference outputs are saved as text for readability in "all_text_out.txt".

At least 6GB of VRAM and 16GB of RAM are required to run the collection.
"""




import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import csv
import glob

MODEL_PATH = "./deepseek-coder-1.3b-instruct"
SAE_DIR = "./LSAE_models"
PROMPTS_FILE = "prompts.txt"

MAX_NEW_TOKENS = 200
NUM_LAYERS = 24
NUM_FEATURES = 4096

NPY_OUT = "all_features_out.npy"
CSV_OUT = "all_tokens_out.csv"
TXT_OUT = "all_text_out.txt"


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model=2048, d_hidden=4096):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        out = self.decoder(h)
        return out, h


def find_sae_path_for_layer(layer_idx: int, sae_dir: str) -> str:
    pattern = os.path.join(sae_dir, f"layer_{layer_idx}_*.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No SAE checkpoint found for layer {layer_idx} with pattern: {pattern}")
    return matches[-1]



if not os.path.exists(PROMPTS_FILE):
    raise FileNotFoundError(f"Prompts file not found: {PROMPTS_FILE}")

with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

num_prompts = len(prompts)
print(f"[INFO] Loaded {num_prompts} prompts\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto" if device.type == "cuda" else None,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
)
model.eval()

print("[STEP] Loading SAEs...")
saes = []
sae_paths = []
for layer_idx in range(NUM_LAYERS):
    p = find_sae_path_for_layer(layer_idx, SAE_DIR)
    sae_paths.append(p)
    sae = SparseAutoencoder()
    state_dict = torch.load(p, map_location="cpu")
    sae.load_state_dict(state_dict)
    sae.eval()
    saes.append(sae)
print("[INFO] SAEs loaded.\n")

hidden_states_collected_by_layer = [[] for _ in range(NUM_LAYERS)]


def make_capture_hook(layer_idx: int):
    def hook_fn(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        last_token_state = t[0, -1, :].detach().cpu().float()
        hidden_states_collected_by_layer[layer_idx].append(last_token_state)
    return hook_fn


hooks = []
for layer_idx in range(NUM_LAYERS):
    h = model.model.layers[layer_idx].mlp.register_forward_hook(make_capture_hook(layer_idx))
    hooks.append(h)

total_features = NUM_LAYERS * NUM_FEATURES
all_prompt_indices = []
all_token_texts = []
all_features = np.zeros((total_features, MAX_NEW_TOKENS * num_prompts), dtype=np.uint8)
all_outputs = []

for p_idx, prompt in enumerate(prompts):
    print(f"\n======== PROMPT {p_idx} ========")
    print(prompt)

    hidden_states_collected_by_layer = [[] for _ in range(NUM_LAYERS)]

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]

    print("[STEP] Generating tokens...")
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            out = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    gen_ids = input_ids[0][encoded["input_ids"].shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    print("\n--- MODEL OUTPUT ---\n")
    print(gen_text)
    print("--------------------\n")

    all_outputs.append(f"\n===== PROMPT {p_idx} =====\n{prompt}\n\n{gen_text}\n")

    total_tokens = len(hidden_states_collected_by_layer[0])
    print(f"[INFO] Collected {total_tokens} hidden states per layer.")

    if total_tokens == 0:
        continue

    tok_count = min(total_tokens, MAX_NEW_TOKENS)

    gen_token_ids = input_ids[0][-tok_count:].tolist()
    tokens = tokenizer.convert_ids_to_tokens(gen_token_ids)
    tokens = [t.replace("▁", " ") for t in tokens]

    all_prompt_indices.extend([p_idx] * tok_count)
    all_token_texts.extend(tokens)

    col0 = p_idx * MAX_NEW_TOKENS
    col1 = col0 + tok_count
    for layer_idx in range(NUM_LAYERS):
        hs_list = hidden_states_collected_by_layer[layer_idx][:tok_count]
        hs = torch.stack(hs_list, dim=0)  # (tok_count, d_model) on CPU F32

        with torch.no_grad():
            _, H = saes[layer_idx](hs)    # (tok_count, 4096)

        H_np = (
            H.clamp(min=0.0, max=2.0)      # (tok_count, 4096)
             .mul(255.0 / 2.0)
             .to(torch.uint8)
             .cpu()
             .numpy()
        )

        base = layer_idx * NUM_FEATURES
        # write rows - features for this layer, cols = tokens for this prompt
        all_features[base:base + NUM_FEATURES, col0:col1] = H_np.T  # (4096, tok_count)
        
for h in hooks:
    h.remove()

print(f"\n[STEP] Writing NPY to {NPY_OUT}...")
np.save(NPY_OUT,all_features)
print(f"\n[STEP] Writing CSV to {CSV_OUT}...")
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_index"] + all_prompt_indices)
    writer.writerow(["token"] + all_token_texts)


print("[DONE] CSV saved.\n")

print(f"[STEP] Writing text outputs to {TXT_OUT}...")
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(all_outputs))

print("[DONE] Output text file saved.\n")
print("ALL DONE.")
