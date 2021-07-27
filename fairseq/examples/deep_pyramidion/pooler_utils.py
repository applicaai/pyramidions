import math
from typing import NamedTuple, Optional, List
from argparse import Namespace
import torch
from torch import nn as nn, Tensor
from .pooler import Pooler
from .successive_halving_topk import TopKConfig, TopKOperator

_supports_blockwise = ['pooler', 'only_blockwise']


def fold_back_maybe(pooler: Pooler, args: Namespace, encoder_out, old_bs):
    """Documents are already chunked, put them back to the old shape (from before pooling)"""
    do_unfold = not pooler.is_lambda and pooler.needs_unfold()

    if do_unfold:
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


def unfold_x_for_blockwise_attention(chunk_size, x):
    """Prepare docs for encoding (split to 'chunks')"""
    old_bs = x.shape[0]
    chunks_num = math.ceil(
        x.shape[1] / chunk_size
    )  # take current batch-document shape

    x_padded = _pad_x(chunk_size, chunks_num, x, old_bs)
    x_padded_unfolded = x_padded.unfold(
        1, chunk_size, chunk_size
    ).flatten(0, 1)

    doc_lengths = (x_padded_unfolded != 0).all(dim=2).sum(axis=1)

    return doc_lengths, x_padded_unfolded, old_bs, chunks_num


def _pad_x(chunk_size, chunks_num, x, old_bs):
    pad_len = (chunks_num * chunk_size) - x.shape[1]
    x_pad = torch.zeros(
        (old_bs, pad_len, x.shape[2]), dtype=x.dtype, device=x.device
    ).fill_(0)
    x_padded = torch.cat(
        (
            x,
            x_pad,
        ),
        dim=1,
    )
    return x_padded


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
        else:
            self.scorer = None

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

        if self.is_lambda:
            return encoder_out
        else:
            encoded_tokens = encoder_out.permute(1, 0, 2)
            bs, input_seq_len, emb_dims = encoded_tokens.shape
            # FIXME: jeżeli są krótsze inputy i jest mniej chunków to to się wyjebie
            if pooled_length == input_seq_len:  # tzn, nie zwróci tutaj!
                return encoder_out

            assert layer_i >= 0 and isinstance(self.scorer, nn.ModuleList)
            token_logits = self.scorer[layer_i](encoded_tokens)

            assert not torch.isnan(token_logits).any()
            assert token_logits.shape[0] == src_lengths.shape[0]
            assert len(token_logits.shape) == 3
            assert token_logits.shape[-1] == 1

            new_token_logits = torch.ones_like(token_logits) * -10000
            for sent_i, slen in enumerate(src_lengths):
                new_token_logits[sent_i, :slen] = token_logits[sent_i, :slen]

            pooled_output, pooled_scores = self.selector(
                encoded_tokens, torch.sigmoid(new_token_logits) + 0.00001
            )
            assert pooled_output.shape[0] == bs
            assert not torch.isnan(pooled_output).any()
            assert pooled_output.shape[1] == self.pooler_config.pooled_len
            return pooled_output.permute(1, 0, 2)
