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
#ifndef MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP
#define MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/common.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Broadcast multiple dimensions between two tensors.
 * Two versions of this operator: 1 input and 2+ inputs.
 * One input version uses output_lens (static target) or output_dyn_dims (symbolic target);
 * see compute_shape for the symbolic single-input contract.
 * 2+ inputs version broadcasts first input to the common shape at evaluation time.
 */
struct multibroadcast
{
    std::vector<std::size_t> output_lens = {};

    // optional attribute
    std::vector<shape::dynamic_dimension> output_dyn_dims = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_lens, "out_lens"), f(self.output_dyn_dims, "out_dyn_dims"));
    }

    std::string name() const { return "multibroadcast"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);

        auto t         = inputs.at(0).type();
        const auto& s0 = inputs.at(0);

        if(s0.ndim() < 1)
        {
            MIGRAPHX_THROW("MULTIBROADCAST: input dimensions should be > 0");
        }

        if(inputs.size() == 1)
        {
            // Symbolic 1-input mode: opt-in via a fully-symbolic output_dyn_dims attribute.
            // Input may be static (bridged via to_symbolic()) or already symbolic.
            // Range-based dynamic input is not allowed.
            const bool symbolic_target = not output_dyn_dims.empty() and
                                         std::all_of(output_dyn_dims.begin(),
                                                     output_dyn_dims.end(),
                                                     [](const auto& d) { return d.is_symbolic(); });

            if(s0.dynamic() and not(symbolic_target and s0.symbolic()))
                MIGRAPHX_THROW(
                    "MULTIBROADCAST: Single dynamic input shape not supported.  Use two inputs.");

            // Shared validation: input dims must align with target dims, with axis-1 broadcast.
            auto validate = [](const auto& in_dims, const auto& out_dims) {
                if(in_dims.size() > out_dims.size())
                    MIGRAPHX_THROW("MULTIBROADCAST: input dimensions should <= output size");
                auto offset = out_dims.size() - in_dims.size();
                for(std::ptrdiff_t i = in_dims.size() - 1; i >= 0; --i)
                {
                    if(out_dims[i + offset] != in_dims[i] and in_dims[i] != 1)
                        MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + to_string_range(in_dims) +
                                       "} cannot be broadcasted to {" + to_string_range(out_dims) +
                                       "}!");
                }
            };

            if(symbolic_target)
            {
                auto s0_sym = s0.to_symbolic();
                validate(s0_sym.dyn_dims(), output_dyn_dims);
                return make_bcast_shape(s0_sym, output_dyn_dims);
            }

            validate(s0.lens(), output_lens);
            return make_bcast_shape(s0, output_lens);
        }
        else
        {
            // 2+ inputs
            if(std::any_of(
                   inputs.cbegin(), inputs.cend(), [](auto input) { return input.dynamic(); }))
            {
                if(not output_dyn_dims.empty())
                {
                    return {t, output_dyn_dims};
                }
                return {t, compute_common_dyn_dims(inputs)};
            }
            else
            {
                // output_lens will not be set for 2+ input version
                auto bcast_lens = compute_common_lens(inputs);
                return make_bcast_shape(s0, bcast_lens);
            }
        }
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        return args[0].reshape(dyn_out.computed_shape);
    }
    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
