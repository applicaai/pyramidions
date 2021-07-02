# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .pyramidion import base_architecture  # noqa

from fairseq.models import register_model_architecture

@register_model_architecture('pyramidion', 'pyramidion_base')
def base_architecture_here(args):
    base_architecture(args)
