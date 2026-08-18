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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_CONVOLUTION_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct module_pass_manager;

/**
 * Transform convolutions layout
 */
struct MIGRAPHX_EXPORT layout_convolution
{
    enum layout_order
    {
        channels_first,
        channels_last,
        channels_auto
    };
    layout_order order = channels_first;
    // Only used with channels_last: convolutions with at least this many
    // output channels store their weights with the K dim innermost (yxck
    // instead of kyxc for 2-D); 1 always applies it, 0 disables it. K-innermost
    // makes the implicit-GEMM A matrix M-contiguous, avoiding power-of-2 row
    // strides.
    std::size_t output_channels_last_threshold = 0;
    // Restrict the output-channels-last weight layout to these types; empty
    // applies to all types.
    std::vector<shape::type_t> output_channels_last_types = {};
    std::string name() const { return "layout_convolution"; }
    void apply(module_pass_manager& mpm) const;
    // Applies this->order, which must be resolved to channels_first or channels_last.
    void apply_layout(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_LAYOUT_CONVOLUTION_HPP
