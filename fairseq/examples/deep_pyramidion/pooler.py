# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class Pooler(nn.Module):
    """Base class for poolers."""

    def forward(self, encoded_tokens, src_lengths=None, **kwargs):
        """
        Args:
            encoded_tokens (LongTensor): encoded tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    @property
    def is_lambda(self):
        return self.args.encoder_pooling == 'lambda'