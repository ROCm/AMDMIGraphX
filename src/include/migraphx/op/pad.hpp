/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/clamp.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pad
{
    std::vector<int64_t> pads;
    float value = 0.0f;
    enum pad_op_mode_t
    {
        constant_pad,
        reflect_pad,
        edge_pad
    };
    pad_op_mode_t mode = constant_pad;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"), f(self.pads, "pads"), f(self.value, "value"));
    }

    std::string name() const { return "pad"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        const auto& s0 = inputs.front();
        if(s0.dynamic())
        {
            auto out_dyn_dims = s0.dyn_dims();
            for(std::size_t i = 0; i < s0.ndim(); ++i)
            {
                out_dyn_dims[i] += pads[i] + pads[i + s0.ndim()];
            }
            return {s0.type(), out_dyn_dims};
        }
        else
        {
            auto&& idims = s0.lens();
            std::vector<std::size_t> rdims(idims.begin(), idims.end());
            std::size_t num_dims = rdims.size();
            for(std::size_t i = 0; i < num_dims; i++)
            {
                rdims[i] += pads[i] + pads[i + num_dims];
            }
            return s0.with_lens(rdims);
        }
    }

    std::size_t pad_ndims() const
    {
        assert(pads.size() % 2 == 0);
        return pads.size() / 2;
    }

    bool symmetric() const
    {
        std::size_t num_dims = pads.size() / 2;
        return std::equal(
            pads.begin(), pads.begin() + num_dims, pads.begin() + num_dims, pads.end());
    }

    // Calculate reflected index using triangle wave formula
    static std::size_t reflect_index(int64_t idx, std::size_t size)
    {
        if(size == 1)
            return 0;

        auto period = size - 1;

        // Handle negative indices by taking absolute value
        auto shifted = idx < 0 ? static_cast<std::size_t>(-idx) : static_cast<std::size_t>(idx);

        // Triangle wave: oscillates between 0 and period
        auto mod_val = shifted % (2 * period);
        return (mod_val <= period) ? mod_val : (2 * period - mod_val);
    }

    // Map output index to input index for reflect/edge padding
    std::size_t map_index(int64_t idx, std::size_t dim_size) const
    {
        if(idx >= 0 and idx < static_cast<int64_t>(dim_size))
            return static_cast<std::size_t>(idx);
        if(mode == op::pad::reflect_pad)
            return reflect_index(idx, dim_size);
        // edge padding: clamp to boundary
        return (idx < 0) ? 0 : dim_size - 1;
    }

    // NOLINTNEXTLINE
    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        assert(dyn_out.computed_shape.standard());
        argument result{dyn_out.computed_shape};
        auto input_shape  = args[0].get_shape();
        auto output_shape = dyn_out.computed_shape;
        auto input_lens   = input_shape.lens();
        auto ndim         = input_lens.size();

        if(mode == op::pad::constant_pad)
        {
            // Constant padding: fill with value, then copy input
            result.visit([&](auto output) {
                using type = typename decltype(output)::value_type;
                std::fill(output.begin(), output.end(), pad_clamp<type>(value));
            });

            visit_all(result, args[0])([&](auto output, auto input) {
                shape_for_each(input_shape, [&](const auto& idx) {
                    std::vector<std::size_t> new_idx(idx.size());
                    std::transform(
                        idx.begin(), idx.end(), pads.begin(), new_idx.begin(), [](auto i, auto j) {
                            return i + j;
                        });
                    output(new_idx.begin(), new_idx.end()) = input(idx.begin(), idx.end());
                });
            });
        }
        else
        {
            // Reflect or edge padding: iterate over output and map to input
            visit_all(result, args[0])([&](auto output, auto input) {
                shape_for_each(output_shape, [&](const auto& out_idx) {
                    std::vector<std::size_t> in_idx(ndim);
                    for(std::size_t d = 0; d < ndim; ++d)
                    {
                        auto idx  = static_cast<int64_t>(out_idx[d]) - pads[d];
                        in_idx[d] = map_index(idx, input_lens[d]);
                    }
                    output(out_idx.begin(), out_idx.end()) = input(in_idx.begin(), in_idx.end());
                });
            });
        }

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
