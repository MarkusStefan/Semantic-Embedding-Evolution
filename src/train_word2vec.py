from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.word2vec_torch import (
    SGNSPairsDataset,
    SkipGramNegSampling,
    UnigramNegativeSampler,
    cosine_drift,
    reservoir_sample_pmi_pairs,
)


@dataclass
class Word2VecTimeSliceConfig:
    vocab_size: int
    embedding_dim: int = 100
    num_negatives: int = 10
    batch_size: int = 4096
    epochs_per_year: int = 1
    lr: float = 2e-3
    weight_decay: float = 0.0
    unigram_power: float = 0.75
    min_pmi: float = 0.0
    max_pairs_per_year: int = 2_000_000
    reservoir_seed: int = 0
    weight_mode: str = "pmi"  # 'pmi' or 'one'
    weight_transform: str = "log1p"  # 'none' or 'log1p'
    warm_start: bool = True
    device: str = "auto"  # 'auto'|'cpu'|'cuda'


@dataclass
class Word2VecTimeSliceResult:
    years: List[int]
    embeddings: List[torch.Tensor]  # list of [V, D] on CPU
    mean_drift: List[float]  # len(years)-1
    checkpoints: List[str]


def _pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unigram_probs(unigram_counts: torch.Tensor, power: float) -> torch.Tensor:
    counts = unigram_counts.float().clone()
    counts[counts < 0] = 0
    probs = torch.pow(counts, float(power))
    s = probs.sum()
    if s <= 0:
        # Fallback to uniform if something goes wrong
        probs = torch.ones_like(probs)
        s = probs.sum()
    return probs / s


def _transform_weights(weights: torch.Tensor, mode: str) -> torch.Tensor:
    w = weights.float()
    if mode == "none":
        return w
    if mode == "log1p":
        return torch.log1p(torch.clamp(w, min=0))
    raise ValueError(f"Unknown weight_transform={mode}")


def train_word2vec_time_slices(
    *,
    years: Sequence[int],
    pmi_csv_pattern: str,
    save_dir: str,
    cfg: Word2VecTimeSliceConfig,
) -> Word2VecTimeSliceResult:
    """Train SGNS Word2Vec independently per year, optionally warm-starting.

    Notes:
      - Uses reservoir sampling to avoid loading huge PMI CSVs into memory.
      - Uses unigram distribution estimated from sampled positives.
    """

    years = [int(y) for y in years]
    device = _pick_device(cfg.device)

    os.makedirs(save_dir, exist_ok=True)

    model = SkipGramNegSampling(cfg.vocab_size, cfg.embedding_dim)
    model.to(device)

    prev_in: Optional[torch.Tensor] = None
    prev_out: Optional[torch.Tensor] = None

    year_embeddings: List[torch.Tensor] = []
    checkpoints: List[str] = []
    mean_drifts: List[float] = []

    for t_idx, year in tqdm(enumerate(years), total=len(years), desc="Years"):
        csv_path = pmi_csv_pattern.format(t_idx=t_idx, year=year)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing PMI file: {csv_path}")

        sampled = reservoir_sample_pmi_pairs(
            csv_path,
            vocab_size=cfg.vocab_size,
            max_pairs=cfg.max_pairs_per_year,
            min_pmi=cfg.min_pmi,
            seed=cfg.reservoir_seed + t_idx,
            weight_mode=cfg.weight_mode,
        )

        weights = _transform_weights(sampled.weight, cfg.weight_transform)
        # Normalize weights so loss scale stays reasonable
        wmean = weights.mean().clamp(min=1e-6)
        weights = weights / wmean

        unigram_probs = _unigram_probs(sampled.unigram_counts, cfg.unigram_power)
        neg_sampler = UnigramNegativeSampler(unigram_probs, device=device)

        dataset = SGNSPairsDataset(sampled.center, sampled.context, weights)
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        if cfg.warm_start and prev_in is not None and prev_out is not None:
            model.in_embed.weight.data.copy_(prev_in.to(device))
            model.out_embed.weight.data.copy_(prev_out.to(device))

        optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

        model.train()
        for _ in range(int(cfg.epochs_per_year)):
            for center, context, w in loader:
                center = center.to(device, non_blocking=True)
                context = context.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)

                neg = neg_sampler.sample(center.shape[0], int(cfg.num_negatives))
                loss = model(center, context, neg, weights=w)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

        model.eval()
        in_emb = model.in_embed.weight.detach().cpu().clone()
        out_emb = model.out_embed.weight.detach().cpu().clone()

        ckpt_path = os.path.join(save_dir, f"sgns_year{year}.pt")
        torch.save(
            {
                "year": year,
                "t_idx": t_idx,
                "in_embed": in_emb,
                "out_embed": out_emb,
                "cfg": cfg.__dict__,
                "csv_path": csv_path,
            },
            ckpt_path,
        )
        checkpoints.append(ckpt_path)
        year_embeddings.append(in_emb)

        if t_idx > 0:
            d = cosine_drift(year_embeddings[-2], year_embeddings[-1])
            mean_drifts.append(float(d.mean().item()))

        prev_in, prev_out = in_emb, out_emb

    return Word2VecTimeSliceResult(
        years=years,
        embeddings=year_embeddings,
        mean_drift=mean_drifts,
        checkpoints=checkpoints,
    )
