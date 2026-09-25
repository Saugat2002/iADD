"""
UNTESTED. Day-1 sanity check for a freshly rented Phase-2 GPU box.

This script has NOT been run against a real LLaDA-8B checkpoint -- it was
written from the LLaDA-8B-Instruct model card / repo conventions as of the
Phase-2 planning pass, before any Phase-2 GPU time was spent. Expect to
debug argument names, generate-call signatures, and trust_remote_code
module paths on first run. Treat every kwarg below as "best guess, verify
against the actual model card / modeling code once loaded."

Purpose: load GSAI-ML/LLaDA-8B-Instruct, generate one completion for a
GSM8K-style math prompt using its native diffusion sampling, and print it.
If this runs and produces coherent (even if wrong) text, the environment
(torch/transformers/trust_remote_code, GPU memory, bf16 support) is sound
and you can move on to the FK steering study. If it fails, fix it here
before touching any of the training/FK scripts.

Usage:
    python smoke_llada.py
"""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

# A GSM8K-style prompt -- swap for an actual GSM8K test-set question once
# this smoke test passes.
PROMPT = (
    "Question: Natalia sold clips to 48 of her friends in April, and then "
    "she sold half as many clips in May. How many clips did Natalia sell "
    "altogether in April and May?\nAnswer:"
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found; this will be extremely slow "
              "for an 8B model. Expected to run on a rented A100-80GB.")

    print(f"Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model {MODEL_ID} (bf16, trust_remote_code=True) ...")
    # NOTE: LLaDA ships custom modeling code and is NOT a standard causal-LM
    # architecture (it's a masked discrete diffusion LM), so AutoModel (not
    # AutoModelForCausalLM) + trust_remote_code is the expected load path.
    # VERIFY against the model card once you have network access to it --
    # the exact class name / generate() signature may differ.
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    print("Tokenizing prompt ...")
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    print("Generating with native diffusion sampling ...")
    # LLaDA's own repo exposes a `generate` helper implementing the
    # block/confidence-based iterative unmasking decode loop (distinct from
    # both autoregressive generate() and the MDLM ddpm-style reverse loop
    # used in Phase 1). The exact call signature (steps, gen_length,
    # block_length, temperature, cfg_scale, remasking strategy, etc.) needs
    # to be confirmed against GSAI-ML/LLaDA's `generate.py` reference
    # implementation -- this is a best-effort placeholder call.
    with torch.no_grad():
        try:
            out = model.generate(
                input_ids,
                steps=128,
                gen_length=128,
                block_length=32,
                temperature=0.0,
            )
        except TypeError as e:
            print(f"model.generate(...) signature mismatch: {e}")
            print("Check GSAI-ML/LLaDA's reference generate.py for the "
                  "correct kwarg names (steps/gen_length/block_length/"
                  "remasking/cfg_scale) and update this call.")
            raise

    completion = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n----- COMPLETION -----")
    print(completion)
    print("-----------------------")


if __name__ == "__main__":
    main()
