"""
This script trains 24 linear sparse autoencoders for feature mapping of the deepseek-coder-1.3b-instruct large langauge model, 
one for the output of each transformer block.

An assortment of python scripts from codeparrot are fed into the deepseek model and used as training data, with deepseek 
internal activations being used to train the autoencoders.

Each autoencoder has an input/output diminsion of 2048 (the embedding diminsion of the llm), with a hidden "feature" layer 
of 4096, for ~16M parameters each, totaling ~384M F32 parameters for all autoencoders.

The script can be stopped and restarted; training will resume from the last checkpoint.

At least 6GB of VRAM and 16GB of RAM are required to run the training.

Trained autoencoders are saved in the /LSAE_models folder.

Token length is limited to 4000 per code sample, to prevent memory overflow, training on 20k code samples will expose the LSAEs 
to ~75M tokens, which is sufficent for feature extraction. Training may take many, multi-hour sessions depending on the GPU.
"""




import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./deepseek-coder-1.3b-instruct"
JSONL_FILE = "training_scripts.json"
MAX_TOKENS = 4000 # critical to prevent memory overflow
NUM_LAYERS = 24
MAX_SNIPPETS = None

CHECKPOINT_EVERY = 10
MODELS_DIR = "LSAE_models"



# sparse autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model=2048, d_hidden=4096):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        out = self.decoder(h)
        return out, h



# device for base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Base model running on:", device)



# load tokenizer, model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    output_hidden_states=True,
)
model.eval()



# create SLAEs, optimizers
saes = {L: SparseAutoencoder().cpu() for L in range(NUM_LAYERS)}
opts = {L: optim.Adam(saes[L].parameters(), lr=1e-3) for L in range(NUM_LAYERS)}

print(f"Initialized {NUM_LAYERS} SAEs on CPU.")


# checkpoint and auto resume logic
def find_existing_checkpoint():
    """
    Returns:
        (found: bool, resume_index: int)
    found = True only if all 24 layer files exist and share the same N.
    resume_index = N (the json line index to skip to)
    """
    if not os.path.isdir(MODELS_DIR):
        return False, 0

    files = [f for f in os.listdir(MODELS_DIR) if f.startswith("layer_") and f.endswith(".pt")]
    if len(files) < NUM_LAYERS:
        return False, 0

    # parse filenames: layer_<L>_<N>.pt
    parsed = []
    for f in files:
        parts = f[:-3].split("_")  # remove .pt
        if len(parts) != 3:
            return False, 0
        try:
            L = int(parts[1])
            N = int(parts[2])
        except ValueError:
            return False, 0
        parsed.append((L, N, f))

    # check all layer indices exist exactly once
    layers_present = set(p[0] for p in parsed)
    if layers_present != set(range(NUM_LAYERS)):
        return False, 0

    # check all N's are the same
    Ns = set(p[1] for p in parsed)
    if len(Ns) != 1:
        return False, 0

    resume_index = Ns.pop()
    return True, resume_index


def load_saes_from_checkpoint(resume_index):
    """
    Load existing layer_x_resume_index.pt into SAEs.
    """
    print(f"\n[Resume] Loading SAEs from checkpoint {resume_index}...\n")
    for L in range(NUM_LAYERS):
        path = os.path.join(MODELS_DIR, f"layer_{L}_{resume_index}.pt")
        state = torch.load(path, map_location="cpu")
        saes[L].load_state_dict(state)



# checkpoint saving
def save_saes(iteration: int):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # delete previous
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith("layer_") and fname.endswith(".pt"):
            try:
                os.remove(os.path.join(MODELS_DIR, fname))
            except OSError:
                pass

    # save new
    for L in range(NUM_LAYERS):
        path = os.path.join(MODELS_DIR, f"layer_{L}_{iteration}.pt")
        torch.save(saes[L].state_dict(), path)

    print(f"[Checkpoint] Saved SAEs at iteration {iteration}.")



# stream json content
def stream_python_snippets(jsonl_path, start_idx=0):
    print(f"\nStreaming JSON-lines from: {jsonl_path}\n")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue  # skip until resume point

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {idx}")
                continue

            content = obj.get("content")
            if not content:
                continue

            fname = obj.get("path", f"snippet_{idx}.py")
            yield idx, fname, content



# train on one code sample
def train_on_snippet(snippet_idx, fname, python_snippet):
    print(f"\n======================================")
    print(f" Snippet {snippet_idx}  |  {fname}")
    print(f"======================================")

    # tokenize
    encoded = tokenizer(
        python_snippet,
        return_tensors="pt",
        truncation=False,
    )

    input_ids = encoded["input_ids"][0]
    total_tokens = len(input_ids)
    print(f"Original token count: {total_tokens}")

    if total_tokens > MAX_TOKENS:
        print(f"Truncating -> {MAX_TOKENS}")
        input_ids = input_ids[:MAX_TOKENS]

    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    print(f"Token count used: {input_ids.shape[1]}")

    # collect activations
    collected = {L: [] for L in range(NUM_LAYERS)}
    handles = []

    for L in range(NUM_LAYERS):
        def make_hook(layer_idx):
            def hook(module, inp, out):
                acts = out.detach().squeeze(0).to("cpu", dtype=torch.float32)
                collected[layer_idx].append(acts)
            return hook

        h = model.model.layers[L].mlp.register_forward_hook(make_hook(L))
        handles.append(h)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    for h in handles:
        h.remove()

    # train SLAEs
    for L in range(NUM_LAYERS):
        if not collected[L]:
            print(f"  Layer {L:02d}: no activations, skipping.")
            continue

        acts = torch.cat(collected[L], dim=0).float()

        sae = saes[L]
        opt = opts[L]

        sae.train()
        opt.zero_grad()

        out, h = sae(acts)
        loss = ((out - acts)**2).mean() + 1e-3 * h.abs().mean()
        loss.backward()
        opt.step()

        print(f"  Layer {L:02d} | loss: {loss.item():.6f}")

        del acts, out, h

    del collected
    torch.cuda.empty_cache()


# main training loop
resume_found, resume_start = find_existing_checkpoint()

if resume_found:
    print(f"\n[Resume] Found full checkpoint set. Resuming at JSON index {resume_start}\n")
    load_saes_from_checkpoint(resume_start)
    num_trained = resume_start
else:
    print("\n[Resume] No valid checkpoint found. Starting from scratch.\n")
    num_trained = 0


for idx, fname, snippet in stream_python_snippets(JSONL_FILE, start_idx=num_trained):
    train_on_snippet(idx, fname, snippet)
    num_trained += 1

    if num_trained % CHECKPOINT_EVERY == 0:
        save_saes(num_trained)

    if MAX_SNIPPETS is not None and num_trained >= MAX_SNIPPETS:
        print(f"\nReached MAX_SNIPPETS = {MAX_SNIPPETS}, stopping.")
        break

print("\nALL TRAINING COMPLETE.")

if num_trained % CHECKPOINT_EVERY != 0 and num_trained > 0:
    save_saes(num_trained)
