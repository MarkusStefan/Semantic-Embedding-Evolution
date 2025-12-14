import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNegSampling(nn.Module):
    """Skip-gram with Negative Sampling (SGNS) implemented in pure PyTorch.

    This is the classic Word2Vec objective:
      -log σ(u_w · v_c) - Σ_k log σ(-u_w · v_{n_k})
    """

    def __init__(self, vocab_size: int, embedding_dim: int, sparse: bool = False):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)

        self.in_embed = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=sparse)
        self.out_embed = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=sparse)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 0.5 / max(1, self.embedding_dim)
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    def forward(
        self,
        center_ids: torch.Tensor,
        pos_context_ids: torch.Tensor,
        neg_context_ids: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SGNS loss.

        Args:
            center_ids: LongTensor [B]
            pos_context_ids: LongTensor [B]
            neg_context_ids: LongTensor [B, K]
            weights: Optional FloatTensor [B] for per-example weighting

        Returns:
            Scalar loss (mean over batch)
        """

        center_vec = self.in_embed(center_ids)  # [B, D]
        pos_vec = self.out_embed(pos_context_ids)  # [B, D]

        pos_score = torch.sum(center_vec * pos_vec, dim=-1)  # [B]
        pos_loss = -F.logsigmoid(pos_score)  # [B]

        neg_vec = self.out_embed(neg_context_ids)  # [B, K, D]
        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)  # [B]

        loss = pos_loss + neg_loss
        if weights is not None:
            loss = loss * weights

        return loss.mean()

    @torch.no_grad()
    def get_input_embeddings(self) -> torch.Tensor:
        return self.in_embed.weight.detach()


class UnigramNegativeSampler:
    """Draw negatives from a unigram distribution using torch.multinomial."""

    def __init__(self, unigram_probs: torch.Tensor, device: Optional[torch.device] = None):
        if unigram_probs.dim() != 1:
            raise ValueError("unigram_probs must be 1D")

        probs = unigram_probs.float().clone()
        probs[probs < 0] = 0
        s = probs.sum()
        if s <= 0:
            raise ValueError("unigram_probs must have positive mass")
        probs /= s

        self.probs = probs.to(device) if device is not None else probs

    def to(self, device: torch.device) -> "UnigramNegativeSampler":
        self.probs = self.probs.to(device)
        return self

    @torch.no_grad()
    def sample(self, batch_size: int, num_negatives: int) -> torch.Tensor:
        # multinomial returns [batch_size*num_negatives]
        samples = torch.multinomial(
            self.probs,
            num_samples=int(batch_size) * int(num_negatives),
            replacement=True,
        )
        return samples.view(int(batch_size), int(num_negatives))


@dataclass
class ReservoirSampledPairs:
    center: torch.Tensor  # Long [N]
    context: torch.Tensor  # Long [N]
    weight: torch.Tensor  # Float [N]
    unigram_counts: torch.Tensor  # Float [V]


def reservoir_sample_pmi_pairs(
    csv_path: str,
    vocab_size: int,
    max_pairs: int,
    min_pmi: float = 0.0,
    seed: int = 0,
    weight_mode: str = "pmi",
) -> ReservoirSampledPairs:
    """Stream a huge (word,context,pmi) CSV and keep a uniform reservoir of up to max_pairs.

    This avoids loading ~100MB+ files into RAM.

    Args:
        csv_path: path to wordPairPMI_*.csv
        vocab_size: vocabulary size
        max_pairs: reservoir capacity
        min_pmi: filter threshold (e.g., 0 keeps only positive PMI)
        seed: RNG seed
        weight_mode: 'pmi' or 'one'
    """

    import csv
    import random

    rng = random.Random(int(seed))
    max_pairs = int(max_pairs)
    if max_pairs <= 0:
        raise ValueError("max_pairs must be > 0")

    center_buf = [0] * max_pairs
    context_buf = [0] * max_pairs
    weight_buf = [0.0] * max_pairs
    kept = 0
    seen = 0

    unigram = torch.zeros(int(vocab_size), dtype=torch.float32)

    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        # Expect header: word,context,pmi but tolerate missing
        for row in r:
            if len(row) < 3:
                continue

            try:
                w = int(row[0])
                c = int(row[1])
                pmi = float(row[2])
            except ValueError:
                continue

            if pmi < float(min_pmi):
                continue

            seen += 1

            if weight_mode == "one":
                wgt = 1.0
            else:
                wgt = float(pmi)

            # Update approximate unigram counts (from positives)
            if 0 <= w < vocab_size:
                unigram[w] += wgt
            if 0 <= c < vocab_size:
                unigram[c] += wgt

            if kept < max_pairs:
                idx = kept
                kept += 1
            else:
                j = rng.randrange(seen)
                if j >= max_pairs:
                    continue
                idx = j

            center_buf[idx] = w
            context_buf[idx] = c
            weight_buf[idx] = wgt

    if kept == 0:
        raise ValueError(f"No pairs kept from {csv_path} (min_pmi={min_pmi})")

    center = torch.tensor(center_buf[:kept], dtype=torch.long)
    context = torch.tensor(context_buf[:kept], dtype=torch.long)
    weight = torch.tensor(weight_buf[:kept], dtype=torch.float32)

    return ReservoirSampledPairs(center=center, context=context, weight=weight, unigram_counts=unigram)


class SGNSPairsDataset(torch.utils.data.Dataset):
    def __init__(self, center: torch.Tensor, context: torch.Tensor, weight: torch.Tensor):
        if center.shape != context.shape or center.shape != weight.shape:
            raise ValueError("center/context/weight must have same shape")
        self.center = center
        self.context = context
        self.weight = weight

    def __len__(self) -> int:
        return int(self.center.numel())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.center[idx], self.context[idx], self.weight[idx]


@torch.no_grad()
def cosine_drift(u_prev: torch.Tensor, u_curr: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-word cosine distance between consecutive embedding matrices."""
    a = F.normalize(u_prev, p=2, dim=1, eps=eps)
    b = F.normalize(u_curr, p=2, dim=1, eps=eps)
    return 1.0 - torch.sum(a * b, dim=1)
