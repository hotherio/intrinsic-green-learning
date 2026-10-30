"""Real data sets of plan v2: no analytic metric, a supervised target, cached under ``IGL_DATA_CACHE``.

Every loader returns a :class:`RealSet` with train / val / test splits standardised per feature on the
training split only (a dead MNIST pixel keeps unit scale), the same tensors for every arm, and the seed
of the split recorded. Nothing here is imported by the smoke tests: the loaders download on first use
(scikit-learn's OpenML mirror, Hugging Face ``datasets`` and ``transformers``) and cache their arrays as
``.npz`` so a second call is a file read.

- :func:`mnist` and :func:`fashion_mnist`: 784 pixels, 10 classes, the standard 60k / 10k split with a
  validation part taken from the training split.
- :func:`dbpedia_minilm`: DBpedia-14 abstracts embedded by ``all-MiniLM-L6-v2`` (mean pooling, unit
  norm, 384-D), 14 classes; chosen over AG News because a linear probe leaves headroom.
- :func:`gpt2_wte`: GPT-2's input embedding table (50 257 x 768); regression target the log unigram
  frequency of each token on WikiText-103, classification target a 4-way string class of the token.
- :func:`gpt2_hidden_states`: the residual stream of GPT-2 small after block ``layer`` on WikiText-103
  text; regression target the model's next-token entropy, classification target the 4-way string class
  of the *next* token (a property of what comes next, not of the token itself).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["CACHE", "RealSet", "dbpedia_minilm", "fashion_mnist", "gpt2_hidden_states", "gpt2_wte", "mnist", "token_class"]

CACHE = Path(os.environ.get("IGL_DATA_CACHE", "~/igl_data_cache")).expanduser()
_SPLITS = ("train", "val", "test")
TOKEN_CLASSES = ("word", "subword", "number", "other")


@dataclass(slots=True)
class RealSet:
    """One real data set, split and standardised."""

    name: str
    task: str  # "classification" | "regression"
    x: dict[str, torch.Tensor]
    y: dict[str, torch.Tensor]  # long labels for classification, float targets for regression
    n_classes: int
    seed: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ambient(self) -> int:
        return int(self.x["train"].shape[1])

    def sizes(self) -> dict[str, int]:
        return {k: int(v.shape[0]) for k, v in self.x.items()}


def _standardise(x: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    mu = x["train"].mean(0, keepdims=True)
    sd = x["train"].std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0  # a constant feature stays constant (zero after centring), it is not blown up
    return {k: torch.from_numpy(((v - mu) / sd).astype(np.float32)) for k, v in x.items()}


def _split_train(n: int, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return perm[n_val:], perm[:n_val]


def _assemble(
    name: str,
    task: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_val: int,
    seed: int,
    n_classes: int,
    extra: dict[str, Any] | None = None,
) -> RealSet:
    tr, va = _split_train(x_tr.shape[0], n_val, seed)
    x = _standardise({"train": x_tr[tr], "val": x_tr[va], "test": x_te})
    y_np = {"train": y_tr[tr], "val": y_tr[va], "test": y_te}
    if task == "classification":
        y = {k: torch.from_numpy(v.astype(np.int64)) for k, v in y_np.items()}
    else:
        mu, sd = float(y_np["train"].mean()), float(y_np["train"].std()) or 1.0
        y = {k: torch.from_numpy(((v - mu) / sd).astype(np.float32)) for k, v in y_np.items()}
    return RealSet(name=name, task=task, x=x, y=y, n_classes=n_classes, seed=seed, extra=extra or {})


def _cached(path: Path, build: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = build()
    np.savez(path, **arrays)
    return arrays


# --------------------------------------------------------------------------------------------- images


def _openml_images(dataset: str) -> dict[str, np.ndarray]:
    from sklearn.datasets import fetch_openml

    d = fetch_openml(dataset, as_frame=False, parser="auto")
    x = np.asarray(d.data, dtype=np.float32) / 255.0
    y = np.asarray(d.target).astype(np.int64)
    # OpenML keeps the canonical order: the first 60 000 rows are the training set, the last 10 000 the test set
    return {"x_train": x[:60_000], "y_train": y[:60_000], "x_test": x[60_000:], "y_test": y[60_000:]}


def mnist(*, n_val: int = 5_000, seed: int = 0, fashion: bool = False) -> RealSet:
    """MNIST (or Fashion-MNIST) pixels in [0, 1], standardised; 55k / 5k / 10k by default."""
    key = "Fashion-MNIST" if fashion else "mnist_784"
    a = _cached(CACHE / "images" / f"{key}.npz", lambda: _openml_images(key))
    return _assemble(
        "fashion-mnist" if fashion else "mnist",
        "classification",
        a["x_train"],
        a["y_train"],
        a["x_test"],
        a["y_test"],
        n_val=n_val,
        seed=seed,
        n_classes=10,
        extra={"source": f"openml:{key}"},
    )


def fashion_mnist(*, n_val: int = 5_000, seed: int = 0) -> RealSet:
    """Fashion-MNIST, the harder twin of MNIST at the same sizes."""
    return mnist(n_val=n_val, seed=seed, fashion=True)


# ------------------------------------------------------------------------------------ sentence embeddings


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _minilm_embed(texts: list[str], *, batch_size: int = 256, max_length: int = 128) -> np.ndarray:
    """Mean-pooled, unit-norm ``all-MiniLM-L6-v2`` embeddings through ``transformers`` (no sentence-transformers)."""
    from transformers import AutoModel, AutoTokenizer

    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    dev = _device()
    model = AutoModel.from_pretrained(name).to(dev).eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tok(
                texts[start : start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            out.append(torch.nn.functional.normalize(e, dim=1).float().cpu().numpy())
    return np.concatenate(out)


def dbpedia_minilm(*, n_train: int = 120_000, n_val: int = 10_000, n_test: int = 20_000, seed: int = 0) -> RealSet:
    """DBpedia-14 abstracts as MiniLM sentence embeddings (384-D), 14 classes."""

    def build() -> dict[str, np.ndarray]:
        from datasets import load_dataset

        ds = load_dataset("fancyzhx/dbpedia_14")
        rng = np.random.default_rng(seed)
        tr_idx = np.sort(rng.permutation(len(ds["train"]))[: n_train + n_val])
        te_idx = np.sort(rng.permutation(len(ds["test"]))[:n_test])
        tr = ds["train"].select(tr_idx.tolist())
        te = ds["test"].select(te_idx.tolist())
        text = lambda r: f"{r['title']}. {r['content']}"  # noqa: E731
        return {
            "x_train": _minilm_embed([text(r) for r in tr]),
            "y_train": np.asarray(tr["label"], dtype=np.int64),
            "x_test": _minilm_embed([text(r) for r in te]),
            "y_test": np.asarray(te["label"], dtype=np.int64),
        }

    a = _cached(CACHE / "text" / f"dbpedia14_minilm_{n_train + n_val}_{n_test}_s{seed}.npz", build)
    return _assemble(
        "dbpedia-minilm",
        "classification",
        a["x_train"],
        a["y_train"],
        a["x_test"],
        a["y_test"],
        n_val=n_val,
        seed=seed,
        n_classes=14,
        extra={"source": "fancyzhx/dbpedia_14 + sentence-transformers/all-MiniLM-L6-v2 (mean pooling, unit norm)"},
    )


# ------------------------------------------------------------------------------------------ GPT-2 tokens


def token_class(token: str) -> int:
    """4-way string class of a GPT-2 token: word (space-prefixed alphabetic), subword (continuation), number, other."""
    body = token[1:] if token.startswith("Ġ") else token
    if body.isalpha():
        return 0 if token.startswith("Ġ") else 1
    if any(c.isdigit() for c in body):
        return 2
    return 3


def _wikitext_ids(split: str, *, max_tokens: int | None = None) -> np.ndarray:
    """WikiText-103 (raw) tokenised by the GPT-2 tokenizer, as one id stream."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ids: list[np.ndarray] = []
    total = 0
    batch: list[str] = []
    for line in ds["text"]:
        if line.strip():
            batch.append(line)
        if len(batch) == 2048:
            for seq in tok(batch)["input_ids"]:
                ids.append(np.asarray(seq, dtype=np.int32))
                total += len(seq)
            batch = []
            if max_tokens is not None and total >= max_tokens:
                break
    if batch and (max_tokens is None or total < max_tokens):
        for seq in tok(batch)["input_ids"]:
            ids.append(np.asarray(seq, dtype=np.int32))
    stream = np.concatenate(ids)
    return stream if max_tokens is None else stream[:max_tokens]


def gpt2_wte(*, target: str = "logfreq", n_val: int = 5_000, n_test: int = 5_000, seed: int = 0) -> RealSet:
    """GPT-2's token embedding table with the log unigram frequency (``target="logfreq"``) or the 4-way token class."""

    def build() -> dict[str, np.ndarray]:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("gpt2")
        wte = model.wte.weight.detach().float().numpy()
        tok = AutoTokenizer.from_pretrained("gpt2")
        stream = _wikitext_ids("train")
        counts = np.bincount(stream, minlength=wte.shape[0]).astype(np.float64)
        logfreq = np.log((counts + 1.0) / (counts.sum() + wte.shape[0]))
        classes = np.asarray([token_class(t) for t in tok.convert_ids_to_tokens(list(range(wte.shape[0])))], dtype=np.int64)
        return {"wte": wte, "logfreq": logfreq.astype(np.float32), "classes": classes, "counts": counts}

    a = _cached(CACHE / "text" / "gpt2_wte_wikitext103.npz", build)
    n = a["wte"].shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    te, tr = perm[:n_test], perm[n_test:]
    if target == "logfreq":
        return _assemble(
            "gpt2-wte-logfreq",
            "regression",
            a["wte"][tr],
            a["logfreq"][tr],
            a["wte"][te],
            a["logfreq"][te],
            n_val=n_val,
            seed=seed,
            n_classes=0,
            extra={"source": "gpt2 wte; unigram counts on wikitext-103-raw-v1 train", "classes": TOKEN_CLASSES},
        )
    if target == "class":
        return _assemble(
            "gpt2-wte-class",
            "classification",
            a["wte"][tr],
            a["classes"][tr],
            a["wte"][te],
            a["classes"][te],
            n_val=n_val,
            seed=seed,
            n_classes=4,
            extra={"source": "gpt2 wte; string class of the token", "classes": TOKEN_CLASSES},
        )
    raise ValueError(f"target must be 'logfreq' or 'class', got {target!r}")


def gpt2_hidden_states(
    *,
    target: str = "entropy",
    n_tokens: int = 50_000,
    layer: int = 6,
    ctx: int = 512,
    n_val: int = 5_000,
    n_test: int = 10_000,
    seed: int = 0,
) -> RealSet:
    """Residual stream of GPT-2 small after block ``layer`` on WikiText-103 validation text (the Mac-sized pilot).

    Positions come from disjoint ``ctx``-long chunks; the first token of every chunk is dropped (no context).
    ``target="entropy"``: the model's next-token entropy in nats at that position; ``target="next-class"``: the
    4-way string class of the next token.
    """

    def build() -> dict[str, np.ndarray]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        dev = _device()
        lm = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()
        stream = _wikitext_ids("validation", max_tokens=n_tokens + ctx)
        n_chunks = math.ceil(n_tokens / (ctx - 1))
        states: list[np.ndarray] = []
        entropy: list[np.ndarray] = []
        next_ids: list[np.ndarray] = []
        with torch.no_grad():
            for c in range(n_chunks):
                chunk = torch.from_numpy(stream[c * ctx : (c + 1) * ctx].astype(np.int64))
                if chunk.shape[0] < ctx:
                    break
                out = lm(chunk.unsqueeze(0).to(dev), output_hidden_states=True)
                h = out.hidden_states[layer + 1][0, 1:-1].float().cpu().numpy()  # positions 1..ctx-2 (next token known)
                logp = torch.log_softmax(out.logits[0, 1:-1].float(), dim=-1)
                states.append(h)
                entropy.append((-(logp.exp() * logp).sum(-1)).cpu().numpy())
                next_ids.append(chunk[2:].numpy())
        classes = np.asarray([token_class(t) for t in tok.convert_ids_to_tokens(list(range(len(tok))))], dtype=np.int64)
        nxt = np.concatenate(next_ids)
        return {
            "states": np.concatenate(states)[:n_tokens],
            "entropy": np.concatenate(entropy)[:n_tokens].astype(np.float32),
            "next_class": classes[nxt][:n_tokens],
        }

    a = _cached(CACHE / "text" / f"gpt2_small_L{layer}_ctx{ctx}_{n_tokens}.npz", build)
    n = a["states"].shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    te, tr = perm[:n_test], perm[n_test:]
    if target == "entropy":
        return _assemble(
            f"gpt2-h{layer}-entropy",
            "regression",
            a["states"][tr],
            a["entropy"][tr],
            a["states"][te],
            a["entropy"][te],
            n_val=n_val,
            seed=seed,
            n_classes=0,
            extra={"source": f"gpt2 hidden_states[{layer + 1}] on wikitext-103 validation; next-token entropy (nats)"},
        )
    if target == "next-class":
        return _assemble(
            f"gpt2-h{layer}-nextclass",
            "classification",
            a["states"][tr],
            a["next_class"][tr],
            a["states"][te],
            a["next_class"][te],
            n_val=n_val,
            seed=seed,
            n_classes=4,
            extra={"source": f"gpt2 hidden_states[{layer + 1}]; string class of the next token", "classes": TOKEN_CLASSES},
        )
    raise ValueError(f"target must be 'entropy' or 'next-class', got {target!r}")


LOADERS: dict[str, Callable[..., RealSet]] = {
    "mnist": mnist,
    "fashion-mnist": fashion_mnist,
    "dbpedia-minilm": dbpedia_minilm,
    "gpt2-wte-logfreq": lambda **kw: gpt2_wte(target="logfreq", **kw),
    "gpt2-wte-class": lambda **kw: gpt2_wte(target="class", **kw),
    "gpt2-h6-entropy": lambda **kw: gpt2_hidden_states(target="entropy", **kw),
    "gpt2-h6-nextclass": lambda **kw: gpt2_hidden_states(target="next-class", **kw),
}
"""Loaders by set name, for the blocks' ``--sets`` option."""
