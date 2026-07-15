/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include <string>
#include <migraphx/op/builder/kit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// Registration point for the torch kit. The tm:: composite builders are defined in
// op/builder/torch/*.cpp and auto-register themselves; this only wires up the passthrough
// ops and the aliases to existing shared builders.
struct torch_kit : kit<torch_kit>
{
    std::string prefix() const { return "tm::"; }
    void apply() const
    {
        this->common_ops({
            "abs",   "acos",        "add",     "asin",    "atan",  "bitwise_and", "ceil", "convert",
            "cos",   "cosh",        "div",     "elu",     "equal", "erf",         "exp",
            "floor", "fmod",        "greater", "isinf",   "isnan", "leaky_relu",  "less", "log",
            "log2",  "logical_and", "max",     "min",     "mul",   "neg",         "not",  "pow",
            "recip", "relu",        "rsqrt",   "sigmoid", "sign",  "sin",         "sinh", "sqrt",
            "sub",   "tan",         "tanh",
        });
        this->common_ops({"where"}, {.common_type = false});

        this->ops({
            "argmax",
            "argmin",
            "broadcast",
            "concat",
            "contiguous",
            "convolution_backwards",
            "dequantizelinear",
            "gather",
            "gathernd",
            "get_tuple_elem",
            "logsoftmax",
            "multibroadcast",
            "pad",
            "pooling",
            "prefix_scan_sum",
            "quantizelinear",
            "reduce_all",
            "reduce_any",
            "reduce_max",
            "reduce_mean",
            "reduce_min",
            "reduce_prod",
            "reduce_sum",
            "reshape",
            "scatter_none",
            "slice",
            "softmax",
            "squeeze",
            "step",
            "topk",
            "transpose",
            "undefined",
            "unsqueeze",
        });

        // Composite builders (bias fusion, broadcasting, etc.), not plain ops.
        this->builders({"batchnorm",
                        "clip",
                        "convolution",
                        "dot",
                        "floor_div",
                        "gather_elements",
                        "gelu_erf",
                        "glu",
                        "group_norm",
                        "hardsigmoid",
                        "instance_norm",
                        "layer_norm",
                        "selu",
                        "softsign",
                        "vector_norm"});
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
