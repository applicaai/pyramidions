import torch
from torch import nn as nn
from .pooler import Pooler
from .successive_halving_topk import TopKConfig, TopKOperator


_supports_blockwise = ['pooler', 'only_blockwise']


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
        self.epsilon = 0.00001


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
        self, encoder_out, layer_i=-1, **kwargs
    ):
        """
        Args:
            encoded_tokens (FloatTensor): encoded tokens in the source language of shape
                `(batch, src_len, emb_dim)`
        """

        if self.is_lambda:
            return encoder_out
        else:
            encoded_tokens = encoder_out.permute(1, 0, 2)
            bs, input_seq_len, emb_dims = encoded_tokens.shape

            if self.selector.pooled_len == input_seq_len:
                return encoder_out

            assert layer_i >= 0 and isinstance(self.scorer, nn.ModuleList)  # FIXME: Remove
            token_logits = self.scorer[layer_i](encoded_tokens)

            assert not torch.isnan(token_logits).any()
            assert token_logits.shape == torch.Size([bs, input_seq_len, 1])

            pooled_output, pooled_scores = self.selector(
                encoded_tokens, torch.sigmoid(token_logits) + self.epsilon
            )
            assert not torch.isnan(pooled_output).any()
            assert pooled_output.shape == torch.Size([bs, self.pooler_config.pooled_len, emb_dims])
            return pooled_output.permute(1, 0, 2)
