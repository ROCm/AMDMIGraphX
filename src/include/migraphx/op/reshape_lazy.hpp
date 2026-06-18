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
#ifndef MIGRAPHX_GUARD_OPERATORS_RESHAPE_LAZY_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESHAPE_LAZY_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/reshape_dims.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reshape_lazy
{
    std::vector<dim_like> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

    value attributes() const { return {{"require_std_shape", true}}; }

    std::string name() const { return "reshape_lazy"; }

    shape dyn_compute_shape(shape s0) const
    {
        const auto& dyn_dims = s0.dyn_dims();
        auto num_not_fixed   = std::count_if(
            dyn_dims.cbegin(), dyn_dims.cend(), [](const auto& dd) { return not dd.is_fixed(); });
        if(num_not_fixed != 1)
        {
            MIGRAPHX_THROW(
                "reshape_lazy: Only supports one non-fixed dynamic_dimension but input {" +
                to_string_range(dyn_dims) + "} has " + to_string(num_not_fixed));
        }
        // track number of fixed elements in input and output
        std::size_t num_dims_ele = 1;
        std::size_t num_dd_ele   = 1;
        for(std::size_t i = 0; i < dyn_dims.size(); ++i)
        {
            if(dyn_dims[i].is_fixed())
            {
                num_dims_ele *= std::get<int64_t>(dims[i]);
                num_dd_ele *= dyn_dims[i].get_interval().min;
            }
            else
            {
                if(dims[i] != dim_like{0} and dims[i] != dim_like{-1})
                {
                    MIGRAPHX_THROW(
                        "reshape_lazy: Non-fixed dynamic_dimension doesn't match with 0 or -1 "
                        "output dimension");
                }
            }
        }
        if(num_dims_ele != num_dd_ele)
        {
            MIGRAPHX_THROW("reshape_lazy: Number of fixed elements must match. Input: " +
                           std::to_string(num_dd_ele) + " Output: " + std::to_string(num_dims_ele));
        }
        // construct output dynamic shape from dims attribute
        std::vector<shape::dynamic_dimension> output_dyn_dims(dims.size());
        std::transform(dims.cbegin(),
                       dims.cend(),
                       dyn_dims.cbegin(),
                       output_dyn_dims.begin(),
                       [](const dim_like& d, auto dyn_dim) {
                           if(not dyn_dim.is_fixed())
                               return dyn_dim;
                           std::size_t dim = std::get<int64_t>(d);
                           return shape::dynamic_dimension{dim, dim};
                       });
        return {s0.type(), output_dyn_dims};
    }

    // Resolves the output layout for static and symbolic input through one path.
    shape symbolic_compute_shape(const shape& s0) const
    {
        // Lift static input to symbolic literals so the same dd arithmetic resolves both.
        auto sym_in          = s0.to_symbolic();
        auto output_dyn_dims = resolve_reshape_dims(sym_in, dims);
        const bool has_inferred_dim =
            std::find(dims.begin(), dims.end(), dim_like{-1}) != dims.end();

        std::vector<sym::expr> target(output_dyn_dims.size());
        std::transform(output_dyn_dims.begin(),
                       output_dyn_dims.end(),
                       target.begin(),
                       [](const auto& dd) { return dd.sym_expr; });

        // Lazy reshape is a no-copy view: when the permutation can't be preserved we
        // cannot fall back to a repacked standard layout the way reshape does.
        auto s = reshape_dims(sym_in, target, {.lazy = true});
        if(not s.has_value())
            MIGRAPHX_THROW("reshape_lazy on axis that is not packed.");

        const bool dims_have_symbolic = std::any_of(dims.begin(), dims.end(), is_symbolic);
        // Only a static input with integer dims is fully literal; evaluate it back to
        // the concrete layout (static results stay byte-identical). Else stays symbolic.
        if(not s0.symbolic() and not dims_have_symbolic)
        {
            auto result = s->to_static();
            if(result.elements() != s0.elements())
                MIGRAPHX_THROW(
                    "reshape_lazy: Wrong number of elements for reshape_lazy: reshape_lazy has " +
                    std::to_string(result.elements()) + " elements whereas the input has " +
                    std::to_string(s0.elements()));
            assert(result.bytes() == s0.bytes());
            return result;
        }
        // A symbolic dim or inferred -1 leaves runtime divisibility to the caller.
        if(not has_inferred_dim and not dims_have_symbolic and
           s->sym_elements() != s0.sym_elements())
            MIGRAPHX_THROW(
                "reshape_lazy: Wrong number of elements for reshape_lazy: reshape_lazy has " +
                to_string(s->sym_elements()) + " elements whereas the input has " +
                to_string(s0.sym_elements()));
        return *s;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);

        validate_reshape_dims(name(), dims);

        const auto& s0 = inputs.front();
        if(s0.dynamic() and not s0.symbolic())
        {
            // A symbolic dim has no range interpretation, so it cannot target a
            // range-based input.
            if(std::any_of(dims.begin(), dims.end(), is_symbolic))
                MIGRAPHX_THROW("reshape_lazy: range-based input only supports int64 dim entries");
            return dyn_compute_shape(s0);
        }
        return symbolic_compute_shape(s0);
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
