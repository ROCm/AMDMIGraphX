# -------------------------------------------------------------------------
# Copyright (c) Advanced Micro Device Inc. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Implements ONNX's backend API.
"""
import sys
if sys.version_info < (3, 0):
    sys.exit()

import migraphx
from onnx.backend.base import BackendRep
import numpy as np
from typing import Any, Tuple


class MIGraphXBackendRep(BackendRep):
    """
    Computes the prediction for a pipeline converted into
    an :class:`onnxruntime.InferenceSession` node.
    """
    def __init__(self, prog, input_names):
        """
        :param session: :class:`migraphx.program`
        """
        self._program = prog
        self._input_names = input_names

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        """
        Computes the prediction.
        See :meth:`migraphx.program.run`.
        """

        if isinstance(inputs, list):
            inps = {}
            for i, name in enumerate(self._input_names):
                inps[name] = migraphx.argument(inputs[i])
            mgx_outputs = self._program.run(inps)
            outs = []
            for out in mgx_outputs:
                outs.append(np.array(out))
            return outs
        else:
            inp = self._program.get_parameter_shapes().keys()
            if len(inp) != 1:
                raise RuntimeError("Model expect {0} inputs".format(len(inp)))
            inps = {inp[0]: migraphx.argument(inputs)}
            mgx_outputs = self._program.run(inps)
            outs = []
            for out in mgx_outputs:
                outs.append(np.array(out))
            return self._program.run(inps)
