# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Implements ONNX's backend API.
"""
import migraphx
from onnx.backend.base import BackendRep
from typing import Any, Tuple


class MIGraphXBackendRep(BackendRep):
    """
    Computes the prediction for a pipeline converted into
    an :class:`onnxruntime.InferenceSession` node.
    """

    def __init__(self, prog):
        """
        :param session: :class:`migraphx.program`
        """
        self._program = prog

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        """
        Computes the prediction.
        See :meth:`migraphx.program.run`.
        """

        if isinstance(inputs, list):
            inps = {}
            for i, name in enumerate(self._program.get_parameter_shapes().keys()):
                inps[name] = inputs[i]
            outs = self._program.run(inps)
            return outs
        else:
            inp = self._program.get_parameter_shapes().keys()
            if len(inp) != 1:
                raise RuntimeError("Model expect {0} inputs".format(len(inp)))
            inps = {inp[0]: inputs}
            return self._program.run(inps)
