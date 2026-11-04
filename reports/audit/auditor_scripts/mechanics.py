"""Read-only: instantiate the submitted model on CPU, count parameters, check mechanics."""
import json, sys
from pathlib import Path
import numpy as np
import torch
root = Path("<repo>")
sys.path.insert(0, str(root / "src"))
from ehrforecast.training import load_config, build_model, collate, model_config
from ehrforecast.tokenizer import FoldTokenizer, age_features
from ehrforecast.model import rotary, EHRTransformer
from ehrforecast.sequences import Sample
from ehrforecast.predict import load_fold_model

torch.manual_seed(0)
config = load_config(root / "configs/experiments/v2_c1.toml")
tok = FoldTokenizer.load(root / "models/transformer/small-v2-base/fold0/tokenizer.json")
codes = [l.split(",")[0] for l in (root/"data/processed/targets.csv").read_text().splitlines()[1:]]
print("n target codes:", len(codes), "vocab:", tok.size)
model = build_model(config, tok, len(codes))
mc = model_config(config, tok, len(codes))
print("ModelConfig:", mc.to_dict())
n = sum(p.numel() for p in model.parameters())
print("fresh model parameters:", n)
print("  token_embedding:", model.token_embedding.weight.numel(), " blocks:", sum(p.numel() for p in model.blocks.parameters()),
      " image_encoder:", sum(p.numel() for p in model.image_encoder.parameters()) if model.image_encoder else 0,
      " classifier:", sum(p.numel() for p in model.classifier.parameters()), " age_embedding:", sum(p.numel() for p in model.age_embedding.parameters()))
# compare with the frozen checkpoint
fm = load_fold_model(root / "models/transformer/small-v2-c1/fold0/F", "best.pt")
state = torch.load(root / "models/transformer/small-v2-c1/fold0/F/best.pt", map_location="cpu", weights_only=False)
print("checkpoint epoch:", state["epoch"], "stage:", state["provenance"]["stage"], "config name:", state["config"]["name"])
print("checkpoint parameters (sum of tensors):", sum(v.numel() for v in state["model"].values()))
print("state dict keys match fresh model:", set(state["model"].keys()) == set(model.state_dict().keys()))
print("output_mask (input-only ids) count:", int(fm.model.output_mask.sum()), "of", fm.model.output_mask.numel())
for cfg_name in ["small-v2-c1","small-v2-c1-seed2","small-v2-c1-seed3"]:
    for fold in range(5):
        st = torch.load(root/"models/transformer"/cfg_name/f"fold{fold}"/"F"/"best.pt", map_location="cpu", weights_only=False)
        print(f"  {cfg_name} fold{fold}: params={sum(v.numel() for v in st['model'].values())} epoch={st['epoch']} classifier.out={st['model']['classifier.bias'].shape[0]} classifier.in={st['model']['classifier.weight'].shape[1]}")

m = fm.model.eval()
d = m.config.d_model
# --- output shape
def mk(tokens, n_prefix=2, readout=None, ages=None):
    t = np.array(tokens, dtype="int64")
    a = np.linspace(30, 40, len(t)).astype("float32") if ages is None else np.asarray(ages, dtype="float32")
    return Sample(tokens=t, ages=a, sources=np.r_[np.full(n_prefix, -1), np.zeros(len(t)-n_prefix)].astype("int64"),
                  n_prefix=n_prefix, readout=len(t)-1 if readout is None else readout,
                  head=np.array([0.5, 1.0, 0.3, 0.0, 0.0], dtype="float32"))
sex_f = tok.token_id("[SEX:F]"); sep = tok.token_id("[SEP]")
ev = [i for i, t in enumerate(tok.vocabulary) if t.startswith(("cond:", "med:", "proc:", "obs:"))][:40]
s1 = mk([sex_f, sep, *ev[:5]]); s2 = mk([sex_f, sep, *ev[5:20]]); s3 = mk([sex_f, sep, *ev[20:23]])
with torch.no_grad():
    b = collate([s1, s2, s3], torch.device("cpu"))
    out = m(b["tokens"], b["ages"], b["readout"], b["head"], None, b["event_mask"])
print("output shape for 3 samples:", tuple(out.shape), "(expect (3, 40))")
# --- causal mask: later tokens do not change earlier states
with torch.no_grad():
    toks = torch.tensor([[sex_f, sep, *ev[:10]]]); ages = torch.from_numpy(age_features(np.linspace(30, 40, 12)[None]))
    changed = toks.clone(); changed[0, 8:] = torch.tensor(ev[30:34])
    h1, h2 = m.hidden(toks, ages), m.hidden(changed, ages)
print("causal: earlier states unchanged:", torch.allclose(h1[:, :8], h2[:, :8], atol=1e-6), "| later states changed:", not torch.allclose(h1[:, 8:], h2[:, 8:]))
# --- padding excluded from readout: single vs batched (right padding)
from ehrforecast.finetune import predict_samples
alone = np.vstack([predict_samples(m, [s], torch.device("cpu"), 1) for s in (s1, s2, s3)])
together = predict_samples(m, [s1, s2, s3], torch.device("cpu"), 3)
print("padding: alone vs batched max abs diff:", float(np.abs(alone - together).max()))
# manual last_mean readout check: mean over event positions only (prefix and padding excluded)
with torch.no_grad():
    b = collate([s1, s2], torch.device("cpu"))
    states = m.hidden(b["tokens"], b["ages"])
    logits = m.classify(states, b["readout"], b["head"], b["event_mask"])
    manual = []
    for i, s in enumerate((s1, s2)):
        L = len(s.tokens); last = states[i, L-1]; mean = states[i, s.n_prefix:L].mean(0)
        manual.append(m.classifier(torch.cat([last, mean, b["head"][i]])))
    manual = torch.stack(manual)
print("last_mean readout equals manual (prefix+padding excluded):", torch.allclose(logits, manual, atol=1e-5))
# --- position encoding: order of events matters (same ages, same multiset of tokens)
with torch.no_grad():
    a = torch.tensor([[sex_f, sep, ev[0], ev[1], ev[2]]]); bb = torch.tensor([[sex_f, sep, ev[1], ev[0], ev[2]]])
    ag = torch.from_numpy(age_features(np.full((1, 5), 35.0)))
    ha, hb = m.hidden(a, ag), m.hidden(bb, ag)
print("RoPE: swapping two event tokens (same ages) changes the final state:", not torch.allclose(ha[0, -1], hb[0, -1], atol=1e-6))
q = torch.randn(1, 1, 6, 8); k = torch.randn(1, 1, 6, 8)
rq, rk = rotary(q, 10000.0), rotary(k, 10000.0)
print("RoPE: position 0 is the identity:", torch.allclose(rq[0,0,0], q[0,0,0]), "| position 3 is rotated:", not torch.allclose(rq[0,0,3], q[0,0,3]))
# shift invariance: score(q at t, k at t+s) depends only on s
qq = q[0,0,0].repeat(6,1)[None,None]; kk = k[0,0,0].repeat(6,1)[None,None]
rq2, rk2 = rotary(qq, 10000.0)[0,0], rotary(kk, 10000.0)[0,0]
s01, s12, s23 = float(rq2[0]@rk2[1]), float(rq2[1]@rk2[2]), float(rq2[2]@rk2[3])
print("RoPE: relative-position scores equal for shift 1 at t=0,1,2:", abs(s01-s12) < 1e-5 and abs(s12-s23) < 1e-5, "| differs for shift 2:", abs(float(rq2[0]@rk2[2]) - s01) > 1e-5)
# --- next-token logits mask input-only symbols
with torch.no_grad():
    lg = m.next_token_logits(torch.randn(2, d))
masked = torch.isinf(lg).all(0)
print("input-only ids masked in next-token logits:", int(masked.sum()), "== input_only_ids", len(tok.input_only_ids()), bool(masked[tok.input_only_ids()].all()))
# --- is dropout 0.3 in the loaded model?
print("dropout p in blocks:", {blk.p for blk in m.blocks}, "head dropout:", m.head_dropout.p, "readout:", m.config.readout, "n_head_features:", m.config.n_head_features)
