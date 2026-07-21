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

    template <class InDims, class OutDims>
    static void validate_broadcast(const InDims& in_dims, const OutDims& out_dims)
    {
        if(in_dims.size() > out_dims.size())
            MIGRAPHX_THROW("MULTIBROADCAST: input dimensions (" + to_string(in_dims.size()) +
                           ") should be <= output size (" + to_string(out_dims.size()) + ")");
        auto offset = out_dims.size() - in_dims.size();
        for(std::ptrdiff_t i = in_dims.size() - 1; i >= 0; --i)
        {
            if(out_dims[i + offset] != in_dims[i] and in_dims[i] != 1)
                MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + to_string_range(in_dims) +
                               "} cannot be broadcasted to {" + to_string_range(out_dims) + "}!");
        }
    }

    static bool is_symbolic_target(const std::vector<shape::dynamic_dimension>& out_dyn_dims)
    {
        return not out_dyn_dims.empty() and
               std::all_of(out_dyn_dims.begin(), out_dyn_dims.end(),
                           [](const auto& d) { return d.is_symbolic(); });
    }

    shape compute_single_input_shape(const shape& s0) const
    {
        // Symbolic 1-input mode: opt-in via a fully-symbolic output_dyn_dims attribute.
        // Input may be static (bridged via to_symbolic()) or already symbolic.
        // Range-based dynamic input is not allowed.
        const bool symbolic_target = is_symbolic_target(output_dyn_dims);
        if(not output_dyn_dims.empty() and not symbolic_target)
            MIGRAPHX_THROW("MULTIBROADCAST: output_dyn_dims must be fully symbolic but given {" +
                           to_string_range(output_dyn_dims) + "}");

        if(s0.dynamic() and not(symbolic_target and s0.symbolic()))
            MIGRAPHX_THROW("MULTIBROADCAST: Single dynamic input shape not supported.  Use two "
                           "inputs. Input shape: " +
                           to_string(s0));

        if(symbolic_target)
        {
            auto s0_sym = s0.to_symbolic();
            validate_broadcast(s0_sym.dyn_dims(), output_dyn_dims);
            return make_bcast_shape(s0_sym, output_dyn_dims);
        }

        validate_broadcast(s0.lens(), output_lens);
        return make_bcast_shape(s0, output_lens);
    }

    shape compute_multi_input_dynamic_shape(shape::type_t t,
                                            const std::vector<shape>& inputs) const
    {
        if(not output_dyn_dims.empty())
        {
            if(not inputs[0].dynamic())
                return {t, output_dyn_dims};
            const auto num_dims        = output_dyn_dims.size();
            const auto num_input_dims  = inputs[0].ndim();
            const auto& input_dyn_dims = inputs[0].dyn_dims();
            std::vector<shape::dynamic_dimension> new_output_dyn_dims(num_dims);
            for(std::size_t i = 0; i < num_dims; ++i)
            {
                if(i < num_input_dims and input_dyn_dims[i].is_symbolic() and
                   not input_dyn_dims[i].is_fixed())
                    new_output_dyn_dims[i] = input_dyn_dims[i];
                else
                    new_output_dyn_dims[i] = output_dyn_dims[i];
            }
            return {t, new_output_dyn_dims};
        }
        return {t, compute_common_dyn_dims(inputs)};
    }

    shape compute_multi_input_static_shape(const shape& s0,
                                           const std::vector<shape>& inputs) const
    {
        // output_lens will not be set for 2+ input version
        if(not output_dyn_dims.empty())
        {
            const auto& in_lens = s0.lens();
            std::vector<std::size_t> bcast_lens(output_dyn_dims.size());
            for(std::size_t i = 0; i < output_dyn_dims.size(); ++i)
            {
                if(output_dyn_dims[i].is_fixed())
                    bcast_lens[i] = shape::static_dim_value(output_dyn_dims[i]);
                else
                    bcast_lens[i] = in_lens[i];
            }
            validate_broadcast(in_lens, bcast_lens);
            return make_bcast_shape(s0, bcast_lens);
        }
        auto bcast_lens = compute_common_lens(inputs);
        return make_bcast_shape(s0, bcast_lens);
    }

    shape compute_multi_input_shape(const shape& s0, const std::vector<shape>& inputs) const
    {
        if(std::any_of(inputs.cbegin(), inputs.cend(), [](auto input) { return input.dynamic(); }))
            return compute_multi_input_dynamic_shape(s0.type(), inputs);
        return compute_multi_input_static_shape(s0, inputs);
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);

        const auto& s0 = inputs.at(0);
        if(s0.ndim() < 1)
        {
            MIGRAPHX_THROW("MULTIBROADCAST: input dimensions should be > 0 but input has rank " +
                           to_string(s0.ndim()));
        }

        if(inputs.size() == 1)
            return compute_single_input_shape(s0);
        return compute_multi_input_shape(s0, inputs);
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
