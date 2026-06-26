"""Tiny end-to-end smoke of the train->export->int8->parity->serve chain to
validate the ST5 / optimum / onnxruntime / tokenizers APIs before the long run.
Runs on ~24 throwaway rows in a temp dir. Not part of the pipeline."""
from __future__ import annotations

import os
import tempfile

os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import onnxruntime as ort
import torch
from datasets import Dataset
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (CrossEncoderTrainer,
                                                 CrossEncoderTrainingArguments)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          set_seed)

set_seed(1234)
rows = [("port for wireguard", "wireguard listens on udp 51820", 1),
        ("port for wireguard", "the focaccia needs an overnight proof", 0)] * 12
ds = Dataset.from_dict({"query": [r[0] for r in rows], "passage": [r[1] for r in rows],
                        "label": [float(r[2]) for r in rows]})
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1, max_length=256)
args = CrossEncoderTrainingArguments(
    output_dir=tempfile.mkdtemp() + "/ck", num_train_epochs=1, per_device_train_batch_size=8,
    learning_rate=2e-5, logging_steps=1, report_to=[], seed=1234, dataloader_num_workers=0,
    eval_strategy="no", save_strategy="no")
CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=BinaryCrossEntropyLoss(model)).train()

d = tempfile.mkdtemp()
mdir, odir = d + "/model", d + "/onnx"
model.save_pretrained(mdir)
os.makedirs(odir, exist_ok=True)
ORTModelForSequenceClassification.from_pretrained(mdir, export=True).save_pretrained(odir)
tok = AutoTokenizer.from_pretrained(mdir)
tok.save_pretrained(odir)
print("onnx dir:", sorted(os.listdir(odir)))
fp32 = os.path.join(odir, "model.onnx")
int8 = os.path.join(odir, "model_int8.onnx")
assert os.path.exists(fp32), "model.onnx missing"
assert os.path.exists(os.path.join(odir, "tokenizer.json")), "tokenizer.json missing"
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)

enc = tok(["port for wireguard"], ["wireguard listens on udp 51820"],
          return_tensors="np", padding=True, truncation=True, max_length=256)
feeds = {k: v.astype(np.int64) for k, v in enc.items()}
tm = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
with torch.no_grad():
    tl = tm(**{k: torch.tensor(v) for k, v in feeds.items()}).logits.numpy().reshape(-1)
sess_inputs = [i.name for i in ort.InferenceSession(fp32).get_inputs()]
print("ONNX input names:", sess_inputs)


def run(p):
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    names = {i.name for i in s.get_inputs()}
    return s.run(None, {k: v for k, v in feeds.items() if k in names})[0].reshape(-1)


print("torch", tl, "onnx-fp32", run(fp32), "onnx-int8", run(int8))

# validate the exact serve-time path: tokenizers pair-encode + int8 session
from tokenizers import Tokenizer
t = Tokenizer.from_file(os.path.join(odir, "tokenizer.json"))
t.enable_truncation(max_length=256)
t.enable_padding()
encs = t.encode_batch([("port for wireguard", "wireguard listens on udp 51820"),
                       ("port for wireguard", "the focaccia needs an overnight proof")])
s = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
names = {i.name for i in s.get_inputs()}
f = {"input_ids": np.asarray([e.ids for e in encs], dtype=np.int64),
     "attention_mask": np.asarray([e.attention_mask for e in encs], dtype=np.int64),
     "token_type_ids": np.asarray([e.type_ids for e in encs], dtype=np.int64)}
f = {k: v for k, v in f.items() if k in names}
out = s.run(None, f)[0].reshape(-1)
probs = 1 / (1 + np.exp(-out))
print("serve-path int8 probs (relevant, irrelevant):", probs)
assert probs[0] > probs[1], "trained-direction sanity (rel>irrel) failed on smoke"
print("SMOKE OK")
