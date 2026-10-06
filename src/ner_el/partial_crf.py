from __future__ import annotations

from typing import List

import torch
from torch import nn


class PartialCRF(nn.Module):
    """Linear-chain CRF with support for partial token labels via allow-mask."""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = int(num_tags)
        self.start_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _log_partition_sequential(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        allow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        neg_inf = torch.finfo(emissions.dtype).min
        alpha = self.start_transitions.unsqueeze(0).expand(batch_size, -1)
        has_started = torch.zeros(batch_size, dtype=torch.bool, device=emissions.device)
        t_eff = int(mask.sum(dim=1).max().item()) if seq_len > 0 else 0

        for t in range(t_eff):
            mask_t = mask[:, t]
            emit_t = emissions[:, t, :]
            trans_next = torch.logsumexp(
                alpha.unsqueeze(2) + self.transitions.unsqueeze(0),
                dim=1,
            ) + emit_t
            start_next = alpha + emit_t
            next_alpha = torch.where(
                has_started.unsqueeze(1),
                trans_next,
                start_next,
            )
            if allow_mask is not None:
                allow_t = allow_mask[:, t, :]
                next_alpha = next_alpha.masked_fill(~allow_t, neg_inf)
            alpha = torch.where(mask_t.unsqueeze(1), next_alpha, alpha)
            has_started = has_started | mask_t

        end_scores = torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)
        return torch.where(has_started, end_scores, torch.zeros_like(end_scores))

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        allow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log-partition via a pairwise tree reduction in the log semiring.

        Same value and gradients as ``_log_partition_sequential`` but needs
        ~log2(T) batched steps instead of T sequential ones. Requires every
        sequence's valid positions to be a prefix (right padding); otherwise
        it falls back to the sequential loop.
        """
        batch_size, seq_len, num_tags = emissions.shape
        if seq_len == 0:
            return emissions.new_zeros(batch_size)
        lengths = mask.long().sum(dim=1)
        prefix = torch.arange(seq_len, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not bool((prefix == mask).all()):
            return self._log_partition_sequential(emissions, mask, allow_mask)
        t_eff = int(lengths.max().item())
        if t_eff == 0:
            return emissions.new_zeros(batch_size)

        neg = -1e4
        emit = emissions[:, :t_eff, :].float()
        if allow_mask is not None:
            emit = emit.masked_fill(~allow_mask[:, :t_eff, :], neg)
        valid = prefix[:, :t_eff]

        has_any = lengths > 0
        v0 = self.start_transitions.float().unsqueeze(0) + emit[:, 0, :]
        if t_eff == 1:
            alpha = v0
        else:
            # M[b, t, i, j] = transitions[i, j] + emit[b, t+1, j]; padding -> log identity
            mats = self.transitions.float().view(1, 1, num_tags, num_tags) + emit[:, 1:, :].unsqueeze(2)
            identity = torch.full((num_tags, num_tags), neg, device=emit.device, dtype=emit.dtype)
            identity.fill_diagonal_(0.0)
            mats = torch.where(valid[:, 1:].view(batch_size, -1, 1, 1), mats, identity.view(1, 1, num_tags, num_tags))
            n = mats.shape[1]
            while n > 1:
                if n % 2 == 1:
                    pad = identity.view(1, 1, num_tags, num_tags).expand(batch_size, 1, -1, -1)
                    mats = torch.cat([mats, pad], dim=1)
                    n += 1
                left, right = mats[:, 0::2], mats[:, 1::2]
                mats = torch.logsumexp(left.unsqueeze(-1) + right.unsqueeze(-3), dim=-2)
                n = mats.shape[1]
            alpha = torch.logsumexp(v0.unsqueeze(-1) + mats[:, 0], dim=1)

        end_scores = torch.logsumexp(alpha + self.end_transitions.float().unsqueeze(0), dim=1)
        return torch.where(has_any, end_scores, torch.zeros_like(end_scores))

    def forward(
        self,
        emissions: torch.Tensor,
        allow_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.bool()
        numerator = self._log_partition(emissions, mask=mask, allow_mask=allow_mask.bool())
        denominator = self._log_partition(emissions, mask=mask, allow_mask=None)
        return -(numerator - denominator).mean()

    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[List[int]]:
        batch_size, seq_len, num_tags = emissions.shape
        mask = attention_mask.bool()
        t_eff = int(mask.sum(dim=1).max().item()) if seq_len > 0 else 0
        history = torch.zeros(
            seq_len,
            batch_size,
            num_tags,
            dtype=torch.long,
            device=emissions.device,
        )

        score = self.start_transitions.unsqueeze(0).expand(batch_size, -1)
        has_started = torch.zeros(batch_size, dtype=torch.bool, device=emissions.device)

        for t in range(t_eff):
            mask_t = mask[:, t]
            emit_t = emissions[:, t, :]
            transition_scores = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_prev_scores, best_prev_tags = transition_scores.max(dim=1)
            history[t] = best_prev_tags

            start_scores = score + emit_t
            continue_scores = best_prev_scores + emit_t
            next_scores = torch.where(
                has_started.unsqueeze(1),
                continue_scores,
                start_scores,
            )
            score = torch.where(mask_t.unsqueeze(1), next_scores, score)
            has_started = has_started | mask_t

        score = score + self.end_transitions.unsqueeze(0)
        best_last_tags = torch.argmax(score, dim=1)

        paths: List[List[int]] = []
        history_cpu = history.detach().cpu().numpy()
        best_last_cpu = best_last_tags.detach().cpu().numpy()
        mask_cpu = mask.detach().cpu().numpy()
        for b in range(batch_size):
            valid_positions = [int(i) for i, valid in enumerate(mask_cpu[b]) if bool(valid)]
            if not valid_positions:
                paths.append([0] * seq_len)
                continue

            current_tag = int(best_last_cpu[b])
            best_path = [current_tag]
            for pos in reversed(valid_positions[1:]):
                current_tag = int(history_cpu[pos, b, current_tag])
                best_path.append(current_tag)
            best_path.reverse()

            padded = [0] * seq_len
            for idx, pos in enumerate(valid_positions):
                padded[int(pos)] = int(best_path[idx])
            paths.append(padded)
        return paths
