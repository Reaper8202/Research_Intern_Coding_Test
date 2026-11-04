"""Read-only interpretation of the submitted transformer on training out-of-fold patients.

Uses the seed-1 members (small-v2-c1, folds 0 to 4) and each fold's held-out training patients,
so official val is not read and no model is changed. Two analyses:

* source ablation: re-predict with one kind of input removed (a token source, the hx summary,
  the age gaps, the head's recency input, or the event order) and score the out-of-fold
  probabilities with the common prior mask;
* token attribution: gradient x input on the token embeddings for a few targets, averaged per
  token over its occurrences, giving the tokens that raise each target's logit.

Writes reports/interpretation/ablation.json, ablation_per_target.csv, attribution_top_tokens.csv.
Run from the repository root: .venv/bin/python scripts/checks/interpretability.py [--cap N]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ehrforecast.contracts import PredictionInputs, load_training_data
from ehrforecast.evaluation import evaluate_predictions
from ehrforecast.events import SOURCE_ORDER
from ehrforecast.finetune import predict_samples
from ehrforecast.folds import fold_patients, load_folds
from ehrforecast.predict import load_fold_model
from ehrforecast.sequences import build_histories
from ehrforecast.training import collate, config_modalities, sequence_config

DATASET, MODELS, OUT = Path("data/processed"), Path("models"), Path("reports/interpretation")
MEMBER = "small-v2-c1"
ATTRIBUTION_TARGETS = {
    "126906006": "neoplasm of prostate", "22298006": "myocardial infarction",
    "723857007": "silent micro-hemorrhage of brain", "230690007": "stroke",
    "64859006": "osteoporosis", "233604007": "pneumonia", "72892002": "normal pregnancy",
    "39848009": "whiplash injury to neck",
}
ABLATE_SOURCES = ("medications", "observations", "procedures", "encounters", "conditions", "immunizations", "careplans")


def ablated(sample, kind, hx_ids, rng):
    s = dataclasses.replace(sample)
    tokens, ages, sources = s.tokens.copy(), s.ages.copy(), s.sources.copy()
    if kind in ABLATE_SOURCES:
        keep = sources != SOURCE_ORDER.index(kind)
        tokens, ages, sources = tokens[keep], ages[keep], sources[keep]
    elif kind == "summary":
        keep = ~np.isin(tokens, hx_ids)
        tokens, ages, sources = tokens[keep], ages[keep], sources[keep]
        s.n_prefix -= int((~keep).sum())
    elif kind == "age_gaps":
        ages = np.full_like(ages, ages[-1])
    elif kind == "recency":
        head = s.head.copy(); head[2] = 0.0; s.head = head
    elif kind == "shuffle_events":
        idx = np.arange(s.n_prefix, len(tokens)); perm = rng.permutation(idx)
        tokens[idx] = tokens[perm]  # events keep their day's age slot, lose their identity order
    s.tokens, s.ages, s.sources = tokens, ages, sources
    s.readout = len(tokens) - 1 if len(tokens) > s.n_prefix else -1
    return s


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=0); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu"); torch.set_num_threads(4)
    data = load_training_data(DATASET)
    cohort, events, prior, labels, codes = data.inputs.cohort, data.inputs.events, data.inputs.prior, data.labels, data.inputs.codes
    assignments = load_folds(DATASET)
    kinds = ["baseline", *ABLATE_SOURCES, "summary", "age_gaps", "recency", "shuffle_events"]
    oof = {k: pd.DataFrame(np.nan, index=labels.index, columns=codes) for k in kinds}
    attr_sum: dict[str, dict[int, float]] = {c: {} for c in ATTRIBUTION_TARGETS}
    attr_n: dict[str, dict[int, int]] = {c: {} for c in ATTRIBUTION_TARGETS}
    token_names = None
    t0 = time.time()
    for fold in range(5):
        fm = load_fold_model(MODELS / "transformer" / MEMBER / f"fold{fold}" / "F")
        model = fm.model.to(device).eval()
        token_names = fm.tokenizer.vocabulary
        _, held = fold_patients(assignments, fold)
        held = labels.index.intersection(held)
        if args.cap:
            held = held[: args.cap]
        inputs = PredictionInputs(cohort.loc[held], events[events["patient_id"].isin(held)], prior.loc[held], codes)
        modalities = config_modalities(DATASET, fm.config)
        hist = build_histories(fm.tokenizer.encode(inputs.events), inputs.cohort, fm.tokenizer, codes, modalities)
        seq = sequence_config(fm.config)
        samples = [hist.prediction_sample(i, seq) for i in range(len(hist))]
        patients = pd.Index(hist.patients)
        hx_ids = np.array([i for i, n in enumerate(token_names) if n.startswith("hx:")])
        rng = np.random.default_rng(20261028 + fold)
        for kind in kinds:
            ss = samples if kind == "baseline" else [ablated(s, kind, hx_ids, rng) for s in samples]
            probs = predict_samples(model, ss, device, 32)
            oof[kind].loc[patients, :] = np.asarray(probs)
        print(f"fold {fold}: ablations done at {time.time()-t0:.0f}s", flush=True)
        # attribution: gradient x input on the token embeddings
        captured: dict[str, torch.Tensor] = {}
        def hook(_m, _i, out):
            out.retain_grad(); captured["emb"] = out
        handle = model.token_embedding.register_forward_hook(hook)
        pad = fm.tokenizer.token_id("[PAD]")
        for start in range(0, len(samples), 32):
            batch = collate(samples[start:start + 32], device)
            logits = model(batch["tokens"], batch["ages"], batch["readout"], batch["head"], batch.get("images"), batch["event_mask"])
            emb = captured["emb"]
            tok = batch["tokens"].numpy()
            for code in ATTRIBUTION_TARGETS:
                j = codes.index(code)
                model.zero_grad(set_to_none=True)
                if emb.grad is not None:
                    emb.grad = None
                logits[:, j].sum().backward(retain_graph=True)
                a = (emb.grad * emb).sum(-1).detach().numpy()
                valid = tok != pad
                ids, vals = tok[valid], a[valid]
                for i, v in zip(ids.tolist(), vals.tolist()):
                    attr_sum[code][i] = attr_sum[code].get(i, 0.0) + v
                    attr_n[code][i] = attr_n[code].get(i, 0) + 1
        handle.remove()
        print(f"fold {fold}: attribution done at {time.time()-t0:.0f}s", flush=True)
    # scores
    rows, per_target = [], {}
    for kind in kinds:
        frame = oof[kind].dropna()
        ev = evaluate_predictions(labels.loc[frame.index], frame, codes, prior=prior.loc[frame.index])
        rows.append({"ablation": kind, "macro_auroc": ev.summary["macro_auroc"], "mAP": ev.summary["mAP"], "n_patients": ev.summary["n_patients"]})
        per_target[kind] = ev.per_target["auroc"]
    table = pd.DataFrame(rows); base = float(table.loc[table.ablation == "baseline", "macro_auroc"].iloc[0])
    table["auroc_change"] = table["macro_auroc"] - base
    table.to_json(OUT / "ablation.json", orient="records", indent=1)
    pt = pd.DataFrame(per_target); pt.to_csv(OUT / "ablation_per_target.csv")
    print(table.round(4).to_string(index=False))
    out = []
    for code, name in ATTRIBUTION_TARGETS.items():
        s = pd.Series({i: attr_sum[code][i] / attr_n[code][i] for i in attr_sum[code] if attr_n[code][i] >= 20})
        for rank, (i, v) in enumerate(s.sort_values(ascending=False).head(12).items(), 1):
            out.append({"code": code, "target": name, "rank": rank, "token": token_names[i], "mean_attribution": v, "occurrences": attr_n[code][i]})
    pd.DataFrame(out).to_csv(OUT / "attribution_top_tokens.csv", index=False)
    print(pd.DataFrame(out).groupby("target").head(5)[["target", "rank", "token", "mean_attribution", "occurrences"]].to_string(index=False))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
