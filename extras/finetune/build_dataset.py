#!/usr/bin/env python3
"""
build_dataset.py: turn CVE JSON 5 records (the CVEProject/cvelistV5 repo) into
training examples for fine-tuning a small chat model, plus a DiscoGPT eval set.

It teaches SKILLS (reading a description), not FACTS (memorizing CVE IDs):
  product  - which vendor and product does this description affect?
  cwe      - which CWE weakness fits this description?
  severity - how severe is it (CVSS base severity)?
The answers come straight from the record's own fields, so they are real labels,
not made-up ones.

Outputs (in --out):
  train.jsonl          {"task", "cve", "messages": [user, assistant]} per line
  val.jsonl            same format; used to measure validation loss while training
  eval_discogpt.jsonl  held-out CVEs as DiscoGPT test cases (Harness > Evals > New)
  stats.json           counts, for the blog

Standard library only. Usage:
  python3 build_dataset.py --src ~/cvelistV5/cves --out data --train 4000
"""
import argparse
import json
import os
import random
import re
import sys

MAX_DESC = 1200          # characters; longer descriptions are skipped (DiscoGPT eval inputs max 2,000)
MIN_DESC = 60

PROMPTS = {
    "product": ("Which vendor and product does this vulnerability affect? "
                "Answer in exactly this format:\nVendor: <vendor>\nProduct: <product>\n\n"
                "Description: {desc}"),
    "cwe": ("Which CWE weakness best fits this vulnerability? "
            "Answer with the CWE ID and its name.\n\nDescription: {desc}"),
    "severity": ("Based on this description, what is the CVSS base severity "
                 "(LOW, MEDIUM, HIGH or CRITICAL)? Answer with the severity and the score.\n\n"
                 "Description: {desc}"),
}
TASKS = list(PROMPTS)


def first_en(items, key="value"):
    for d in items or []:
        if str(d.get("lang", "")).lower().startswith("en") and d.get(key):
            return " ".join(str(d[key]).split())
    return ""


def clean_name(s):
    s = " ".join(str(s or "").split())
    return "" if s.lower() in ("", "n/a", "na", "unknown", "-", "*") else s[:80]


def parse_record(path):
    """One CVE JSON 5 file -> a compact dict, or None if it isn't usable."""
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    meta = r.get("cveMetadata") or {}
    if meta.get("state") != "PUBLISHED":
        return None
    cna = (r.get("containers") or {}).get("cna") or {}
    desc = first_en(cna.get("descriptions"))
    if not (MIN_DESC <= len(desc) <= MAX_DESC):
        return None
    if desc.lower().startswith("** reject") or desc.lower().startswith("** disputed"):
        return None

    rec = {"cve": meta.get("cveId", ""), "desc": desc}

    # vendor / product: the first affected entry with real names
    for a in cna.get("affected") or []:
        v, p = clean_name(a.get("vendor")), clean_name(a.get("product"))
        if v and p:
            rec["vendor"], rec["product"] = v, p
            break

    # CWE: the first problem type with a real CWE ID
    for pt in cna.get("problemTypes") or []:
        for d in pt.get("descriptions") or []:
            cid = str(d.get("cweId") or "")
            if re.fullmatch(r"CWE-\d{1,5}", cid):
                name = " ".join(str(d.get("description") or "").split())
                name = re.sub(r"^CWE-\d+\s*[:\-]?\s*", "", name)[:120]
                rec["cwe"], rec["cwe_name"] = cid, name
                break
        if "cwe" in rec:
            break

    # severity: CVSS from the CNA first, then from ADP containers (e.g. CISA)
    containers = [cna] + list((r.get("containers") or {}).get("adp") or [])
    for c in containers:
        for m in c.get("metrics") or []:
            for k in ("cvssV3_1", "cvssV3_0", "cvssV4_0"):
                cv = m.get(k)
                if isinstance(cv, dict) and cv.get("baseSeverity") and cv.get("baseScore") is not None:
                    sev = str(cv["baseSeverity"]).upper()
                    if sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
                        rec["severity"], rec["score"] = sev, cv["baseScore"]
                        rec["cvss_version"] = {"cvssV3_1": "3.1", "cvssV3_0": "3.0", "cvssV4_0": "4.0"}[k]
                        break
            if "severity" in rec:
                break
        if "severity" in rec:
            break
    return rec


def available(rec):
    out = []
    if rec.get("product"):
        out.append("product")
    if rec.get("cwe"):
        out.append("cwe")
    if rec.get("severity"):
        out.append("severity")
    return out


def answer(task, rec):
    if task == "product":
        return f"Vendor: {rec['vendor']}\nProduct: {rec['product']}"
    if task == "cwe":
        return f"{rec['cwe']}: {rec['cwe_name']}" if rec.get("cwe_name") else rec["cwe"]
    return f"{rec['severity']} (CVSS {rec['cvss_version']} base score {rec['score']})"


def example(task, rec):
    return {"task": task, "cve": rec["cve"], "messages": [
        {"role": "user", "content": PROMPTS[task].format(desc=rec["desc"])},
        {"role": "assistant", "content": answer(task, rec)},
    ]}


def eval_case(task, rec, n):
    case = {"id": f"{task}-{n}", "input": PROMPTS[task].format(desc=rec["desc"]), "note": rec["cve"]}
    if task == "product":
        case["contains"] = [rec["product"]]    # official vendor names ("Samsung Mobile") often aren't in the text
    elif task == "cwe":
        case["contains"] = [rec["cwe"]]
    else:
        case["contains"] = [rec["severity"]]
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with CVE JSON 5 files (e.g. cvelistV5/cves)")
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--train", type=int, default=4000, help="training examples")
    ap.add_argument("--val", type=int, default=200, help="validation examples")
    ap.add_argument("--eval", type=int, default=30, help="DiscoGPT eval cases (10 per task)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    print(f"scanning {a.src} ...", flush=True)
    paths = []
    for root, _dirs, files in os.walk(a.src):
        paths += [os.path.join(root, f) for f in files if f.startswith("CVE-") and f.endswith(".json")]
    if not paths:
        sys.exit(f"no CVE-*.json files found under {a.src}")
    rng.shuffle(paths)
    print(f"found {len(paths):,} CVE files; reading a random sample", flush=True)

    per_task_eval = max(1, a.eval // len(TASKS))
    need = a.train + a.val
    train_val, evals = [], {t: [] for t in TASKS}
    read = usable = 0
    for p in paths:
        rec = parse_record(p)
        read += 1
        if read % 20000 == 0:
            print(f"  read {read:,}, usable {usable:,}", flush=True)
        if not rec:
            continue
        tasks = available(rec)
        if not tasks:
            continue
        usable += 1
        # Fill the eval set first, one CVE per case, so these CVEs never appear in training.
        open_tasks = [t for t in tasks if len(evals[t]) < per_task_eval]
        if open_tasks:
            t = rng.choice(open_tasks)
            evals[t].append(eval_case(t, rec, len(evals[t]) + 1))
            continue
        train_val.append(example(rng.choice(tasks), rec))   # one task per CVE: more CVEs, more variety
        if len(train_val) >= need:
            break

    if len(train_val) < need:
        print(f"note: only {len(train_val)} usable examples (asked for {need})", flush=True)
    rng.shuffle(train_val)
    val, train = train_val[:a.val], train_val[a.val:]

    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "eval_discogpt.jsonl"), "w", encoding="utf-8") as f:
        f.write("# Held-out CVEs (never in training). Paste into DiscoGPT: Harness > Evals > New.\n")
        f.write("# Run with temp 0.2: base model vs fine-tuned model.\n")
        for t in TASKS:
            for c in evals[t]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    count = lambda rows: {t: sum(1 for r in rows if r["task"] == t) for t in TASKS}
    stats = {"files_found": len(paths), "files_read": read, "usable": usable,
             "train": len(train), "val": len(val), "eval": sum(len(v) for v in evals.values()),
             "train_by_task": count(train), "val_by_task": count(val)}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {a.out}/train.jsonl, val.jsonl, eval_discogpt.jsonl, stats.json")
    print("sample:\n" + json.dumps(train[0]["messages"], indent=2, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
