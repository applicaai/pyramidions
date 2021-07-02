
import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial


class TopKConfig:
    input_len: int = -1
    pooled_len: int = -1
    depth: int = 0
    flip_right: bool = True
    sort_back: bool = False
    base: int = 20

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TopKOperator(nn.Module):
    """Funny no-net implementation xD"""

    def __init__(self):
        super(TopKOperator, self).__init__()
        self.iterations_performed = 0

    def set_config(self, pooler_config):
        self.input_len = pooler_config.input_len
        self.pooled_len = pooler_config.pooled_len
        self.depth = pooler_config.depth if pooler_config.depth > 0 \
            else int(
            torch.log2(torch.tensor(self.input_len / self.pooled_len)))
        self.flip_right = pooler_config.flip_right
        self.sort_back = pooler_config.sort_back
        self.base = pooler_config.base

    def forward(self, embs, scores):
        """
        embs: batch x input_len x emb_depth
        scores: batch x input_len x 1
        """
        new_embs = []
        new_scores = []
        # pad embeddings with zeros
        embs, scores = self.pad_to_input_len(self.input_len, embs, scores)

        for batch_i in range(embs.shape[0]):
            embs_tmp, scores_tmp = self.our_topk(embs[batch_i].unsqueeze(0), scores[batch_i].unsqueeze(0))
            assert len(embs_tmp.shape) == 3 and embs_tmp.shape[0] == 1
            assert len(scores_tmp.shape) == 2 and scores_tmp.shape[0] == 1

            new_embs.append(embs_tmp)
            new_scores.append(scores_tmp)
        new_embs = torch.cat(new_embs, dim=0)
        new_scores = torch.cat(new_scores, dim=0)

        return new_embs, new_scores

    @staticmethod
    def pad_to_input_len(input_len, embs, scores):
        sh = list(embs.shape)
        sh[1] = input_len - sh[1]
        assert sh[1] >= 0
        emb_pad = torch.zeros(sh, dtype=embs.dtype, device=embs.device)
        embs = torch.cat((embs, emb_pad), dim=1)
        # pad scores with negative big score
        sh = list(scores.shape)
        sh[1] = input_len - sh[1]
        score_pad = torch.zeros(sh, dtype=scores.dtype, device=scores.device) + 0.00001
        scores = torch.cat((scores, score_pad), dim=1).squeeze(2)
        return embs, scores

    def our_topk(self, embs, scores):
        """This is an implementation of our topk function"""
        e = embs.shape[2]
        s = partial(F.softmax, dim=1)
        target_size = self.input_len // 2
        for layer in range(self.depth):
            pairs_idx = self.get_topk_pair_idx(scores)      # firstly, sort by scores and 'draw' pairs
            scores_before = scores.clone()
            scores_converged = scores[:, pairs_idx]
            if self.base > 0:
                exped = torch.pow(self.base, scores_converged)      # exponentiation with any given base
                scores_converged = s(exped)                         # softmax over scores (the more it converges usually the better)
            else:
                raise ValueError
            scores = (scores_before[:, pairs_idx] * scores_converged)\
                .sum(dim=1)    # new scores are a linear interpolation in pairs provided
            embs = (embs[:, pairs_idx] * scores_converged.unsqueeze(3)
                    .expand((1, 2, target_size, e)))\
                .sum(dim=1)    # new embedding are also linearly interpolated from the old pair elements

            # De-sort back into chunk-positions
            # (this may be useful if we want to have an old ordering
            # of top-k elements in the sequence)
            if self.sort_back:
                scores = scores[:, pairs_idx[0].sort().indices]
                embs = embs[:, pairs_idx[0].sort().indices]

            # Finish the round with new target assignments
            current_size = target_size
            target_size = embs.shape[1] // 2

            if current_size < self.pooled_len:
                break
        return embs, scores

    def get_topk_pair_idx(self, scores):
        """ Sort by value and fold.
        This is halving the number of inputs in each step.
        This keeps topk token in different sampling 'pool'
        """
        sort_idx = scores.sort(descending=True).indices

        l_half = sort_idx.shape[-1] // 2
        left = sort_idx[:, :l_half]
        right = sort_idx[:, l_half:]
        if self.flip_right:
            right = torch.flip(right, dims=(1, 0))
        pairs_idx = torch.cat((left, right),
                              dim=0)
        return pairs_idx