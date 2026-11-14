#!/usr/bin/env python3
"""
train_smol.py: full fine-tune of a small chat model (default SmolLM2-135M-Instruct)
on the examples from build_dataset.py, on CPU.

"Full" means every weight is trained (like cve-gpt), not a LoRA add-on.
The loss is only counted on the ANSWER tokens, so the model learns to answer,
not to repeat the question.

What it writes (in --out):
  final/           the trained model + tokenizer (Hugging Face format, ready for GGUF conversion)
  checkpoint/      saved every --save-every steps and on Ctrl+C (resume-friendly safety net)
  metrics.jsonl    one line per log/eval step: step, train loss, val loss, time; for the blog chart
  samples.txt      a few answers before and after training, side by side

Needs: torch + transformers (pip install transformers).
Usage (inside tmux):
  python3 train_smol.py --train data/train.jsonl --val data/val.jsonl --out runs/smol135 --limit 300   # smoke test
  nice -n 10 python3 train_smol.py --train data/train.jsonl --val data/val.jsonl --out runs/smol135    # real run

Memory: training keeps every layer's intermediate results for each example until the
backward pass. --batch 2 --accum 4 learns from batches of 8 at about a quarter of the memory
(the default). --grad-ckpt saves more memory still, for ~30% more time.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def ids_of(x):
    # apply_chat_template returns a list of ids, or a dict/BatchEncoding in some versions
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    return list(x)


def encode(tok, ex, max_len):
    """Token ids + labels: -100 (ignored) on the prompt, real ids on the answer."""
    msgs = ex["messages"]
    prompt = ids_of(tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True))
    full = ids_of(tok.apply_chat_template(msgs, tokenize=True))
    if full[:len(prompt)] != prompt:          # template quirk: fall back to training on everything
        labels = list(full)
    else:
        labels = [-100] * len(prompt) + full[len(prompt):]
    if len(full) > max_len:                   # skip rather than cut off the answer
        return None
    return {"input_ids": full, "labels": labels}


def batches(items, size, pad_id, shuffle, rng):
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [items[j] for j in order[i:i + size]]
        n = max(len(c["input_ids"]) for c in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), n), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, c in enumerate(chunk):
            L = len(c["input_ids"])
            ids[k, :L] = torch.tensor(c["input_ids"])
            lab[k, :L] = torch.tensor(c["labels"])
            att[k, :L] = 1
        yield ids, att, lab


@torch.no_grad()
def val_loss(model, data, batch, pad_id):
    model.eval()
    total, n = 0.0, 0
    for ids, att, lab in batches(data, batch, pad_id, False, None):
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        total += out.loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def answer(model, tok, msgs, max_new=60):
    model.eval()
    ids = torch.tensor([ids_of(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))])
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.pad_token_id)
    model.train()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def save(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="runs/smol135")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=2, help="examples per forward/backward pass (memory)")
    ap.add_argument("--accum", type=int, default=4, help="passes per model update; batch x accum = effective batch")
    ap.add_argument("--grad-ckpt", action="store_true", help="recompute layers in the backward pass: less memory, slower")
    ap.add_argument("--max-len", type=int, default=384, help="tokens; longer examples are skipped")
    ap.add_argument("--warmup", type=int, default=30, help="steps of learning-rate warm-up")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training examples (smoke test)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    print(f"torch {torch.__version__}, {a.threads} threads, model {a.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {params / 1e6:.1f}M (cve-gpt v4: 10.7M)", flush=True)

    raw_train, raw_val = load_jsonl(a.train, a.limit), load_jsonl(a.val)
    train = [e for e in (encode(tok, ex, a.max_len) for ex in raw_train) if e]
    val = [e for e in (encode(tok, ex, a.max_len) for ex in raw_val) if e][:200]
    print(f"examples: train {len(train)} (skipped {len(raw_train) - len(train)} too long), val {len(val)}", flush=True)
    if not train:
        sys.exit("no training examples left; raise --max-len")

    print(f"batch {a.batch} x accum {a.accum} = {a.batch * a.accum} examples per update"
          + (", gradient checkpointing on" if a.grad_ckpt else ""), flush=True)
    steps_per_epoch = math.ceil(len(train) / (a.batch * a.accum))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    def lr_at(step):    # linear warm-up, then cosine down to 10%
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        p = (step - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    metrics = open(os.path.join(a.out, "metrics.jsonl"), "a", encoding="utf-8")

    def log(**kv):
        metrics.write(json.dumps(kv) + "\n")
        metrics.flush()

    # "Before" picture: validation loss and a few answers from the untouched model.
    probes = [ex["messages"][:-1] for ex in raw_val[:3]]
    truths = [ex["messages"][-1]["content"] for ex in raw_val[:3]]
    before = [answer(model, tok, m) for m in probes]
    v0 = val_loss(model, val, a.batch, tok.pad_token_id)
    print(f"step 0: val loss {v0:.4f} (untrained)", flush=True)
    log(step=0, val_loss=round(v0, 4), elapsed=0)

    t0, step, run_loss, run_n, done, micro = time.time(), 0, 0.0, 0, False, 0
    try:
        while not done:
            for ids, att, lab in batches(train, a.batch, tok.pad_token_id, True, rng):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                (loss / a.accum).backward()          # gradients add up over `accum` small passes
                run_loss += loss.item()
                run_n += 1
                micro += 1
                if micro % a.accum:
                    continue
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0 or step == total_steps:
                    el = time.time() - t0
                    eta = el / step * (total_steps - step)
                    tl = run_loss / run_n
                    print(f"step {step}/{total_steps}  loss {tl:.4f}  lr {lr_at(step):.2e}  "
                          f"{el / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)
                    log(step=step, train_loss=round(tl, 4), lr=lr_at(step), elapsed=round(el, 1))
                    run_loss, run_n = 0.0, 0
                if step % a.eval_every == 0 or step == total_steps:
                    v = val_loss(model, val, a.batch, tok.pad_token_id)
                    print(f"step {step}: val loss {v:.4f}", flush=True)
                    log(step=step, val_loss=round(v, 4), elapsed=round(time.time() - t0, 1))
                if step % a.save_every == 0:
                    save(model, tok, os.path.join(a.out, "checkpoint"))
                if step >= total_steps:
                    done = True
                    break
    except KeyboardInterrupt:
        print("\nstopped by Ctrl+C: saving what we have to checkpoint/", flush=True)
        save(model, tok, os.path.join(a.out, "checkpoint"))
        sys.exit(1)

    save(model, tok, os.path.join(a.out, "final"))
    after = [answer(model, tok, m) for m in probes]
    with open(os.path.join(a.out, "samples.txt"), "w", encoding="utf-8") as f:
        for m, b, af, t in zip(probes, before, after, truths):
            f.write("QUESTION:\n" + m[-1]["content"] + "\n\nBEFORE:\n" + b + "\n\nAFTER:\n" + af +
                    "\n\nCORRECT:\n" + t + "\n\n" + "=" * 60 + "\n\n")
    mins = (time.time() - t0) / 60
    print(f"\ndone: {step} steps in {mins:.1f} min. Model: {a.out}/final  Samples: {a.out}/samples.txt", flush=True)


if __name__ == "__main__":
    main()
