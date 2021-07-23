import math
from typing import NamedTuple, Optional, List

import torch
from torch import nn as nn, Tensor
from .pooler import Pooler
from .successive_halving_topk import TopKConfig, TopKOperator


def fold_back_maybe(pooler, args, encoder_out, old_bs):
    """Documents are already chunked, put them back to the old shape (from before pooling)"""
    is_blockwise = getattr(pooler.args, "use_sparse_attn", "none") == "only_blockwise"
    stop_blockwise = (
        getattr(pooler.args, "use_sparse_attn", "none") == "pooler_no_block"
    )
    if (not pooler.is_lambda and not stop_blockwise) or is_blockwise:
        # combine chunks back to original batches,
        # but with shorter length (pooled in the 1st dim)
        # this can be achieved by tiling the tensors
        hidden_dim = encoder_out.shape[2]
        enc = encoder_out.shape
        assert enc[1] % old_bs == 0
        assert enc[2] == args.encoder_embed_dim

        encoder_out = (
            encoder_out.transpose(0, 1)
            .reshape([old_bs, -1, hidden_dim])
            .transpose(0, 1)
        )

    return encoder_out


def unfold_maybe(pooler, chunk_size, encoder, doc_lengths, doc_tokens):
    """Prepare docs for encoding (split to 'chunks')"""

    old_bs = doc_tokens.shape[0]
    chunks_num = 1
    is_blockwise = getattr(pooler.args, "use_sparse_attn", "none") == "only_blockwise"
    stop_blockwise = (
        getattr(pooler.args, "use_sparse_attn", "none") == "pooler_no_block"
    )
    do_unfold = (not pooler.is_lambda and not stop_blockwise) or is_blockwise
    if do_unfold:
        chunks_num = math.ceil(
            doc_tokens.shape[1] / chunk_size
        )  # take current batch-document shape

        doc_tokens_padded = _pad(
            chunk_size, chunks_num, doc_tokens, encoder, old_bs
        )
        doc_tokens_unfolded = doc_tokens_padded.unfold(
            1, chunk_size, chunk_size
        ).flatten(0, 1)

        doc_lengths = (doc_tokens_unfolded != 1).sum(axis=1)
        _inject_eos(doc_lengths, doc_tokens_unfolded, encoder)
    else:
        doc_tokens_unfolded = doc_tokens

    return doc_lengths, doc_tokens_unfolded, old_bs, chunks_num


def _inject_eos(doc_lengths, doc_tokens_unfolded, encoder):
    new_bsz = doc_tokens_unfolded.shape[0]
    ind = (
        torch.arange(new_bsz, dtype=torch.long, device=doc_lengths.device),
        doc_lengths - 1,
    )
    doc_tokens_unfolded.index_put_(
        ind,
        torch.full(
            (new_bsz,),
            encoder.dictionary.eos_index,
            device=doc_lengths.device,
            dtype=torch.long,
        ),
    )


def _pad(chunk_size, chunks_num, doc_tokens, encoder, old_bs):
    pad_len = (chunks_num * chunk_size) - doc_tokens.shape[1]
    doc_padder = torch.zeros(
        (old_bs, pad_len), dtype=doc_tokens.dtype, device=doc_tokens.device
    ).fill_(encoder.dictionary.pad_index)
    doc_tokens_padded = torch.cat(
        (
            doc_tokens,
            doc_padder,
        ),
        dim=1,
    )
    return doc_tokens_padded


class TopkPooler(Pooler):
    """
    Token Pooler.

    Args:
        args (configargparse.Namespace): parsed command-line arguments

    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._prepare_pooler()

    def _prepare_pooler(self):
        if self.args.encoder_pooling != "lambda":
            self._set_scorer_architecture()

            self._set_softselector_method()
            # self.bias = nn.Parameter(torch.ones((1, self.args.max_source_positions, 1)))
        else:
            self.scorer = None
            # self.bias = None        # FIXME remove

    def _set_softselector_method(self):
        if self.args.encoder_pooling == "topk":
            self.selector = TopKOperator()
            self.pooler_config = TopKConfig(
                input_len=self.args.max_source_positions,
                pooled_len=None,  # Not known ahead, will be set dynamically
                flip_right=self.args.flip_right,
                base=20,
                hard_topk_inference=False,
            )

    def _set_scorer_architecture(self):
        if self.args.encoder_pooling_arch == "linear":
            self.scorer = nn.ModuleList(
                [
                    nn.Linear(self.args.encoder_embed_dim, 1)
                    for el in range(0, self.args.encoder_layers)
                ]
            )
        else:
            self.scorer = None

    def forward(
        self, encoder_out, src_lengths=None, pooled_length=None, layer_i=-1, **kwargs
    ):
        """
        Args:
            encoded_tokens (FloatTensor): encoded tokens in the source language of shape
                `(batch, src_len, emb_dim)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """

        if self.args.encoder_pooling == "lambda":
            return encoder_out
        else:
            encoded_tokens = encoder_out.permute(1, 0, 2)
            bs, input_seq_len, emb_dims = encoded_tokens.shape

            if pooled_length == input_seq_len:
                return encoder_out

            assert layer_i >= 0 and isinstance(self.scorer, nn.ModuleList)
            token_logits = self.scorer[layer_i](encoded_tokens)

            assert not torch.isnan(token_logits).any()
            assert token_logits.shape[0] == src_lengths.shape[0]
            assert len(token_logits.shape) == 3
            assert token_logits.shape[-1] == 1

            # if self.bias is not None:
            #     token_logits += self.bias[:, :input_seq_len]

            new_token_logits = torch.ones_like(token_logits) * -10000
            for sent_i, slen in enumerate(src_lengths):
                # sent[slen:] = -10000
                new_token_logits[sent_i, :slen] = token_logits[sent_i, :slen]

            if self.args.encoder_pooling == "topk":
                pooled_output, pooled_scores = self.selector(
                    encoded_tokens, torch.sigmoid(new_token_logits) + 0.00001
                )
                assert pooled_output.shape[0] == bs
                assert not torch.isnan(pooled_output).any()
                assert pooled_output.shape[1] == self.pooler_config.pooled_len
                return pooled_output.permute(1, 0, 2)


class FFN(nn.Module):
    def __init__(self, embed_dim, out_dim=1):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
