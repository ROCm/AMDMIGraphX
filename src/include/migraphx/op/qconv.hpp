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
 */
#ifndef MIGRAPHX_GUARD_OPERATORS_QCONV_HPP
#define MIGRAPHX_GUARD_OPERATORS_QCONV_HPP

#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Weight-only quantized convolution for the DxGML weight-only quantization (WoQ) pattern.
 *
 * inputs[0] : activation  — fp16 or bf16  (NCHW or NHWC)
 * inputs[1] : weight      — int8 or uint4 (quantized, stored as-is)
 * inputs[2] : weight scale — fp16 or bf16 (per-output-channel or block-quantized)
 * inputs[3] : weight zero-point — same type as inputs[1], optional
 *
 * Output type matches the activation type (fp16 or bf16).
 *
 * The dequant formula  scale * (weight - zp)  is applied INSIDE the GPU kernel;
 * the activation tensor is used as-is without any pre-dequantization.
 *
 * Lowered to rocMLIR which generates an in-kernel dequant+convolution kernel.
 * Convolution attributes (padding, stride, dilation, group) mirror those of
 * the standard convolution op and are carried through for lowering.
 */
struct qconv
{
    std::vector<std::size_t> padding  = {0, 0};
    std::vector<std::size_t> stride   = {1, 1};
    std::vector<std::size_t> dilation = {1, 1};

    padding_mode_t padding_mode = default_;
    int group                   = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.group, "group"));
    }

    value attributes() const
    {
        return {{"general_data_type", "convolution"}, {"normalize_padding", "padding"}};
    }

    std::string name() const { return "qconv"; }

    void check_attribute_size() const
    {
        if((padding.size() != stride.size() and (padding.size() / 2) != stride.size()) or
           stride.size() != dilation.size())
        {
            MIGRAPHX_THROW("QCONV: inconsistent attribute sizes");
        }
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
    }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3, 4);
        check_attribute_size();

        const shape& act     = inputs[0]; // fp16/bf16 activation
        const shape& weights = inputs[1]; // int8/uint4 weight

        if(act.lens().size() < 3)
            MIGRAPHX_THROW("QCONV: activation must be >= 3D (N + spatial dims + channels)");
        if(weights.lens().size() < 3)
            MIGRAPHX_THROW("QCONV: weight must be >= 3D");

        size_t loc_kdims = act.lens().size() - 2;
        if(loc_kdims != kdims())
            MIGRAPHX_THROW("QCONV: input k-dims does not match attribute size");

        // Output spatial dims computed from standard convolution formula
        std::vector<size_t> output_lens{act.lens()[0], weights.lens()[0]};
        auto padding_size = padding.size();
        for(size_t i = 0; i < loc_kdims; i++)
        {
            auto padding_factor = 2 * padding[i];
            if(padding_size == 2 * loc_kdims)
                padding_factor = padding[i] + padding[i + loc_kdims];
            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(
                1,
                (static_cast<std::ptrdiff_t>(act.lens()[i + 2]) -
                 (1 + static_cast<std::ptrdiff_t>(dilation[i]) *
                          (static_cast<std::ptrdiff_t>(weights.lens()[i + 2]) - 1)) +
                 static_cast<std::ptrdiff_t>(padding_factor)) /
                        static_cast<std::ptrdiff_t>(stride[i]) +
                    1)));
        }

        // Output type matches activation type (fp16 or bf16)
        return {act.type(), output_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_OPERATORS_QCONV_HPP
