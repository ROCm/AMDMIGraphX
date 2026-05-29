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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_MLSS_CONV_OP_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_MLSS_CONV_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/shape.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

enum class mlss_activation_mode : uint8_t
{
    identity    = 0,
    leaky_relu  = 1,
    sigmoid     = 2,
    scaled_tanh = 3,
    relu        = 4,
};

// Intermediate op inserted by fuse_mlss. Carries conv metadata for the JIT
// compiler (jit/mlss_conv.cpp) which converts it into a code_object_op.
struct mlss_conv_op
{
    std::vector<std::size_t> padding{};
    std::vector<std::size_t> stride{1, 1};
    std::vector<std::size_t> dilation{1, 1};
    std::size_t group       = 1;
    bool has_bias           = false;
    uint8_t activation_mode = static_cast<uint8_t>(mlss_activation_mode::identity);
    float activation_alpha  = 0.0f;
    shape output{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.group, "group"),
                    f(self.has_bias, "has_bias"),
                    f(self.activation_mode, "activation_mode"),
                    f(self.activation_alpha, "activation_alpha"),
                    f(self.output, "output"));
    }

    std::string name() const { return "gpu::mlss_conv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        std::size_t expected = has_bias ? 3 : 2;
        if(inputs.size() > expected)
            return inputs.back();
        return output;
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>& inputs) const
    {
        std::size_t expected = has_bias ? 3 : 2;
        if(inputs.size() > expected)
            return {inputs.size() - 1};
        return {};
    }
};

// Binary info returned by the AMDMLSS API query.
struct mlss_conv_binary_info
{
    value::binary code_object{};
    std::string symbol_name{};
    std::size_t n_groups   = 64;
    std::size_t block_size = 256;

    bool empty() const { return code_object.empty(); }
};

#ifdef MIGRAPHX_USE_AMDMLSS
// Query the AMDMLSS API for a non-relocatable conv kernel binary.
// Returns empty info if no kernel is available for the given configuration.
mlss_conv_binary_info query_mlss_conv_binary(const context& ctx,
                                             const std::vector<std::size_t>& act_lens,
                                             const std::vector<std::size_t>& wt_lens,
                                             const std::vector<std::size_t>& out_lens,
                                             const std::vector<std::size_t>& padding,
                                             const std::vector<std::size_t>& stride,
                                             const std::vector<std::size_t>& dilation,
                                             std::size_t group,
                                             bool has_bias_flag,
                                             uint8_t act_mode,
                                             shape::type_t dtype);
#endif

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
