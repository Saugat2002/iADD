"""
Evaluation harness for iADD-LM Phase 1 (reward vs diversity).

For a given backbone (base model or a trained checkpoint from trace_grpo),
samples G continuations per prompt and reports:
  - mean/std reward (same scorer as training)
  - distinct-1/2/3  (unique n-grams / total n-grams, per prompt, averaged)
  - self-BLEU       (avg BLEU of each sample against the other G-1; lower = more diverse)
  - 1 - mean pairwise Jaccard over unigram sets (higher = more diverse)
Writes one JSON line; meant to be called per checkpoint to build the
reward-vs-diversity trajectory.

  python ~/dllm/iadd-lm/eval.py --ckpt none --reward sentiment --group 8
  python ~/dllm/iadd-lm/eval.py --ckpt runs/<run>/backbone_it300.pt ...
"""
import argparse
import collections
import itertools
import json
import math
import os
import sys

import torch

FK_DIR = os.path.expanduser('~/dllm/Fk-Diffusion-Steering/discrete_diffusion')
sys.path.insert(0, FK_DIR)
sys.path.insert(0, os.path.join(FK_DIR, 'mdlm'))
sys.path.insert(0, os.path.expanduser('~/dllm/iadd-lm'))
os.chdir(FK_DIR)

import dataloader  # noqa: E402
from fk_diffusion import compute_rewards  # noqa: E402
from trace_grpo import TraceGRPO, build_config  # noqa: E402


# ---------------- diversity metrics (dependency-free) ----------------
def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts, n):
    all_ng = []
    for t in texts:
        all_ng += ngrams(t.split(), n)
    return len(set(all_ng)) / max(len(all_ng), 1)


def bleu(hyp, refs, max_n=4):
    """Plain corpus-free sentence BLEU with uniform weights + brevity penalty."""
    hyp_tokens = hyp.split()
    if not hyp_tokens:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        h = collections.Counter(ngrams(hyp_tokens, n))
        if not h:
            return 0.0
        max_ref = collections.Counter()
        for r in refs:
            rc = collections.Counter(ngrams(r.split(), n))
            for g, c in rc.items():
                max_ref[g] = max(max_ref[g], c)
        clipped = sum(min(c, max_ref[g]) for g, c in h.items())
        p_n = clipped / sum(h.values())
        if p_n == 0:
            p_n = 1e-9
        log_p += math.log(p_n) / max_n
    ref_len = min(len(r.split()) for r in refs)
    bp = 1.0 if len(hyp_tokens) > ref_len else math.exp(1 - ref_len / len(hyp_tokens))
    return bp * math.exp(log_p)


def self_bleu(texts):
    if len(texts) < 2:
        return 0.0
    scores = [bleu(t, texts[:i] + texts[i + 1:]) for i, t in enumerate(texts)]
    return sum(scores) / len(scores)


def jaccard_diversity(texts):
    sets = [set(t.split()) for t in texts]
    dists = []
    for a, b in itertools.combinations(sets, 2):
        inter, union = len(a & b), len(a | b)
        dists.append(1 - inter / max(union, 1))
    return sum(dists) / max(len(dists), 1)


# ---------------- semantic diversity (embedding-based) ----------------
_SEM_MODEL_CACHE = {}
_SEM_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def _load_semantic_model():
    """Lazily load a sentence-embedding model into a module-level cache.
    Prefers sentence_transformers if importable, else falls back to plain
    transformers (AutoModel + mean pooling). Returns None on failure.
    """
    if 'model' in _SEM_MODEL_CACHE:
        return _SEM_MODEL_CACHE['model']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            _SEM_MODEL_CACHE['model'] = ('st', st_model, None)
        except ImportError:
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(_SEM_MODEL_NAME)
            mdl = AutoModel.from_pretrained(_SEM_MODEL_NAME).to(device)
            mdl.eval()
            _SEM_MODEL_CACHE['model'] = ('hf', mdl, tok)
    except Exception as e:  # noqa: BLE001 - never crash eval over this
        print(f"[warn] semantic_div: failed to load embedding model ({e}); "
              f"semantic_div will be reported as None", file=sys.stderr)
        _SEM_MODEL_CACHE['model'] = None
    return _SEM_MODEL_CACHE['model']


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def semantic_div(texts):
    """Mean pairwise cosine DISTANCE between sentence embeddings of `texts`.
    Higher = more semantically diverse. Returns None if no embedding model
    is available (never raises).
    """
    if len(texts) < 2:
        return 0.0

    cached = _load_semantic_model()
    if cached is None:
        return None
    kind, model, tok = cached

    try:
        with torch.no_grad():
            if kind == 'st':
                embs = model.encode(texts, convert_to_tensor=True,
                                     normalize_embeddings=False)
            else:
                device = next(model.parameters()).device
                batch = tok(texts, return_tensors='pt', padding=True,
                            truncation=True, max_length=256).to(device)
                out = model(**batch)
                embs = _mean_pool(out.last_hidden_state, batch['attention_mask'])
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            sims = embs @ embs.T
            n = sims.shape[0]
            iu = torch.triu_indices(n, n, offset=1)
            pair_sims = sims[iu[0], iu[1]]
            dists = 1.0 - pair_sims
            return float(dists.mean())
    except Exception as e:  # noqa: BLE001 - never crash eval over this
        print(f"[warn] semantic_div: embedding failed ({e}); "
              f"semantic_div will be reported as None", file=sys.stderr)
        return None


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='none', help="'none' for base model or path to backbone_itN.pt")
    ap.add_argument('--checkpoint', default='~/dllm/mdlm-owt-local')
    ap.add_argument('--reward', default='sentiment')
    ap.add_argument('--reward-label', default='positive')
    ap.add_argument('--reward-trim', type=int, default=100)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--gen-len', type=int, default=128)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--prompt-file', default=os.path.join(
        FK_DIR, 'evaluation', 'pplm_discrim_prompts_orig.jsonl'))
    ap.add_argument('--out', default=os.path.expanduser('~/dllm/iadd-lm/evals.jsonl'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--eta', type=float, default=None,
                     help='reward threshold for rare-sample fraction; if unset, '
                          'rare_frac_eta/rare_frac fields are omitted')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = TraceGRPO(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None
    if args.ckpt != 'none':
        ckpt_path = os.path.expanduser(args.ckpt)
        if not os.path.isabs(ckpt_path):  # cwd was chdir-ed to FK_DIR at import
            ckpt_path = os.path.join(os.path.expanduser("~/dllm/iadd-lm"), ckpt_path)
        sd = torch.load(ckpt_path, map_location='cuda')
        model.backbone.load_state_dict(sd)
    model.backbone.eval()

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f]

    per_prompt = []
    for p in prompts:
        enc = tokenizer([p], return_tensors='pt', padding=False)
        prompt_ids = enc['input_ids'][:, :-1].to(model.device)
        with torch.no_grad():
            tr = model.rollout(prompt_ids, args.group, args.steps)
        trim = args.reward_trim + prompt_ids.shape[1]
        texts = tokenizer.batch_decode(tr['final'][:, :trim])
        # strip the prompt so diversity measures only the continuations
        conts = [t.replace('<|endoftext|>', ' ').strip() for t in texts]
        rs = compute_rewards(samples=texts, reward_name=args.reward,
                             reward_label=args.reward_label)
        r_samples = [float(r) for r in rs]
        pp_rec = dict(
            prompt=p,
            r_mean=float(torch.tensor(rs).float().mean()),
            distinct1=distinct_n(conts, 1),
            distinct2=distinct_n(conts, 2),
            distinct3=distinct_n(conts, 3),
            self_bleu=self_bleu(conts),
            jaccard_div=jaccard_diversity(conts),
            semantic_div=semantic_div(conts),
            r_samples=r_samples,
        )
        if args.eta is not None:
            pp_rec['rare_frac'] = sum(1 for r in r_samples if r >= args.eta) / max(len(r_samples), 1)
        per_prompt.append(pp_rec)

    def avg(k):
        vals = [d[k] for d in per_prompt if d[k] is not None]
        return sum(vals) / len(vals) if vals else None

    rec = dict(tag=args.tag, ckpt=args.ckpt, reward=args.reward,
               group=args.group, steps=args.steps, seed=args.seed,
               r_mean=avg('r_mean'), distinct1=avg('distinct1'),
               distinct2=avg('distinct2'), distinct3=avg('distinct3'),
               self_bleu=avg('self_bleu'), jaccard_div=avg('jaccard_div'),
               semantic_div=avg('semantic_div'),
               per_prompt=per_prompt)
    if args.eta is not None:
        all_samples = [r for d in per_prompt for r in d['r_samples']]
        rec['rare_frac_eta'] = (sum(1 for r in all_samples if r >= args.eta)
                                 / max(len(all_samples), 1))
    with open(os.path.expanduser(args.out), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    brief = {k: round(v, 4) for k, v in rec.items()
             if isinstance(v, float)}
    print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()
