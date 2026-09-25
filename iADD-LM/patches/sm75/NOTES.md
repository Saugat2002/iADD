# sm75 (pre-Ampere) patches

These are snapshots of two files that were modified on the `ajad` server to
make the MDLM/FK-diffusion-steering stack run **without flash-attn**, because
that GPU is a pre-Ampere (sm75-class) card that flash-attn does not support.
They replace flash-attn calls with PyTorch's built-in scaled-dot-product
attention (SDPA), which is numerically fine but somewhat slower.

**You only need these if your GPU is pre-Ampere (compute capability < 8.0,
e.g. T4/sm75, V100/sm70).** On Ampere or newer (A100/sm80, H100/sm90),
flash-attn installs and works normally — use the unmodified upstream files
instead and skip this patch entirely.

## Where each file goes

- `dit.py` → replaces
  `Fk-Diffusion-Steering/discrete_diffusion/mdlm/models/dit.py` in your
  clone of the Fk-Diffusion-Steering repo (the DiT-style backbone used by
  MDLM). This is where the flash-attn attention call is swapped for
  `torch.nn.functional.scaled_dot_product_attention`.

- `modeling_mdlm.py` → replaces `modeling_mdlm.py` inside your local clone
  of the base model, i.e. `~/dllm/mdlm-owt-local/modeling_mdlm.py` (a local
  checkout of `kuleshov-group/mdlm-owt`'s modeling file, patched the same
  way).

## How to apply

```bash
cp patches/sm75/dit.py \
   ~/dllm/Fk-Diffusion-Steering/discrete_diffusion/mdlm/models/dit.py

cp patches/sm75/modeling_mdlm.py \
   ~/dllm/mdlm-owt-local/modeling_mdlm.py
```

Diff against the upstream versions of these files first if you're not sure
whether upstream has since changed independently of the attention backend —
these are full-file snapshots, not unified diffs, so a blind copy will
clobber any unrelated upstream changes.
