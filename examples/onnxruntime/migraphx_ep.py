#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
# Shared helper to make the MIGraphX execution provider usable across ONNX Runtime
# versions that register it as a built-in EP as well as newer versions that ship it
# as a plugin EP library that must be registered before use.
import glob
import os

import onnxruntime

MIGRAPHX_EP = "MIGraphXExecutionProvider"


def _find_migraphx_plugin_lib():
    # The plugin EP library (built with --use_migraphx) is shipped inside the
    # onnxruntime package. Allow an explicit override for non-wheel installs.
    override = os.environ.get("MIGRAPHX_EP_LIB")
    if override:
        return override

    package_dir = os.path.dirname(os.path.abspath(onnxruntime.__file__))
    patterns = (
        "capi/libonnxruntime_providers_migraphx.*",
        "libonnxruntime_providers_migraphx.*",
    )
    for pattern in patterns:
        matches = glob.glob(os.path.join(package_dir, pattern))
        if matches:
            return matches[0]
    return None


def _ensure_migraphx_available(registration_name):
    if MIGRAPHX_EP in onnxruntime.get_available_providers():
        return

    if hasattr(onnxruntime, "register_execution_provider_library"):
        lib_path = _find_migraphx_plugin_lib()
        if lib_path is None:
            raise RuntimeError(
                "MIGraphX plugin EP library not found; set MIGRAPHX_EP_LIB to its path"
            )
        onnxruntime.register_execution_provider_library(registration_name,
                                                        lib_path)
        if MIGRAPHX_EP not in onnxruntime.get_available_providers():
            raise RuntimeError(
                f"Registered MIGraphX plugin EP from '{lib_path}' but "
                f"{MIGRAPHX_EP} is still unavailable")
        return

    raise RuntimeError(
        f"This onnxruntime ({onnxruntime.__version__}) has neither a built-in "
        f"MIGraphX EP nor the plugin registration API")


def ensure_migraphx_ep(provider_options=None, registration_name=MIGRAPHX_EP):
    """Ensure the MIGraphX execution provider is available and return the provider
    list to pass to onnxruntime.InferenceSession.

    Works on both classic builds (MIGraphX registered as a built-in EP) and newer
    plugin-EP builds where the EP library must be registered first.

    provider_options: optional dict of MIGraphX EP options (e.g.
    {"migraphx_fp16_enable": "1"}). When provided, the EP is returned as a
    (name, options) tuple so the options are applied to the session directly
    instead of via environment variables.
    """
    _ensure_migraphx_available(registration_name)

    if provider_options:
        return [(MIGRAPHX_EP, provider_options)]
    return [MIGRAPHX_EP]
