/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_GATHERND_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHERND_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct gathernd
{
    int batch_dims = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.batch_dims, "batch_dims"));
    }

    std::string name() const { return "gathernd"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        if(migraphx::none_of(inputs, [](auto v) { return v.dynamic(); }))
        {
            auto r = inputs.front().lens().size();
            auto q = inputs.back().lens().size();
            auto k = inputs.back().lens().back();
            if(k > r - batch_dims)
            {
                MIGRAPHX_THROW("GATHERND: Indices of length " + std::to_string(k) +
                               " cannot be used to access data of rank " +
                               std::to_string(r - batch_dims));
            }
            if(batch_dims >= q or batch_dims >= r)
            {
                MIGRAPHX_THROW("GATHERND: rank of an input cannot be less than batch_dims=" +
                               std::to_string(batch_dims));
            }
            auto indices_lens_iter = inputs.back().lens().begin();
            int output_lens_size   = int(q) + r - k - batch_dims - 1;
            if(output_lens_size <= 0)
            {
                MIGRAPHX_THROW("GATHERND: Indices too large for static data input: k=" +
                               std::to_string(k));
            }
            std::vector<std::size_t> output_lens(output_lens_size);
            std::copy(indices_lens_iter, indices_lens_iter + (q - 1), output_lens.begin());
            if(k < r - batch_dims)
            {
                auto data_lens = inputs.front().lens();
                std::copy(data_lens.begin() + batch_dims + k,
                          data_lens.end(),
                          output_lens.begin() + q - 1);
            }
            shape output_shape{inputs.front().type(), output_lens};
            return output_shape;
        }
        else
        {
            // If one or both inputs are dynamic shapes, the output is dynamic
            auto i_shape    = inputs.back();
            auto data_shape = inputs.front();
            size_t k;
            if(i_shape.dynamic())
            {
                // the rank of the output is a function of k, so it must be fixed.
                // MIGraphX supports dynamic dimensions, but not dynamic NUMBER of dimensions.
                if(!i_shape.dyn_dims().back().is_fixed())
                {
                    MIGRAPHX_THROW(
                        "GATHERND: last dimension of indices tensor must be fixed (min=max)");
                }
                k = i_shape.dyn_dims().back().min;
            }
            else
            {
                k = i_shape.lens().back();
            }
            auto r               = data_shape.ndim();
            auto q               = i_shape.ndim();
            int output_dims_size = int(q) + r - k - batch_dims - 1;
            if(k > r - batch_dims)
            {
                MIGRAPHX_THROW("GATHERND: Indices of length " + std::to_string(k) +
                               " cannot be used to access data of rank " +
                               std::to_string(r - batch_dims));
            }
            if(batch_dims >= q || batch_dims >= r)
            {
                MIGRAPHX_THROW("GATHERND: rank of an input cannot be less than batch_dims=" +
                               std::to_string(batch_dims));
            }
            if(output_dims_size <= 0)
            {
                MIGRAPHX_THROW("GATHERND: Indices too large for data input: k=" +
                               std::to_string(k));
            }
            // 3 vectors that will be used as constructor arguments for the output shape
            std::vector<size_t> mins(output_dims_size);
            std::vector<size_t> maxes(output_dims_size);
            std::vector<size_t> opts(output_dims_size);

            // Part of the output shape comes from indices tensor, part from data tensor
            if(!i_shape.dynamic())
            {
                const auto& indices_lens_iter = i_shape.lens().begin();
                std::copy(indices_lens_iter, indices_lens_iter + (q - 1), mins.begin());
                std::copy(indices_lens_iter, indices_lens_iter + (q - 1), maxes.begin());
                std::fill(opts.begin(), opts.begin() + (q - 1), size_t(0));
            }
            else
            {
                for(size_t i = 0; i < q; i++)
                {
                    const shape::dynamic_dimension& dd = i_shape.dyn_dims().at(i);
                    mins[i]                            = dd.min;
                    maxes[i]                           = dd.max;
                    opts[i]                            = dd.opt;
                }
            }

            // populate from data input
            if(k < r - batch_dims)
            {
                if(!data_shape.dynamic())
                {
                    auto data_lens = data_shape.lens();
                    std::copy(data_lens.begin() + batch_dims + k - 1,
                              data_lens.end(),
                              mins.begin() + q - 1);
                    std::copy(data_lens.begin() + batch_dims + k - 1,
                              data_lens.end(),
                              maxes.begin() + q - 1);
                    std::fill(opts.begin() + q, opts.end(), size_t(0));
                }
                else
                {
                    size_t j;
                    for(size_t i = batch_dims + k, j = q - 1; i < opts.size(); i++, j++)
                    {
                        shape::dynamic_dimension dd = data_shape.dyn_dims()[i];
                        mins[j]                     = dd.min;
                        maxes[j]                    = dd.max;
                        opts[j]                     = dd.opt;
                    }
                }
            }
            migraphx::shape output_shape(inputs.front().type(), mins, maxes, opts);
            return output_shape;
        }
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        visit_all(result, args[0])([&](auto output, auto data) {
            args[1].visit([&](auto indices) {
                auto indices_shape        = indices.get_shape();
                auto indices_shape_lens   = indices_shape.lens();
                auto data_shape           = data.get_shape();
                auto data_shape_lens      = data_shape.lens();
                auto k                    = indices_shape.lens().back();
                const auto num_slice_dims = k;
                std::size_t num_slices    = std::accumulate(indices_shape_lens.begin(),
                                                         indices_shape_lens.end() - 1,
                                                         1,
                                                         std::multiplies<std::size_t>());
                std::size_t slice_size  = std::accumulate(data_shape_lens.begin() + k + batch_dims,
                                                         data_shape_lens.end(),
                                                         1,
                                                         std::multiplies<std::size_t>());
                std::size_t num_batches = std::accumulate(data_shape_lens.begin(),
                                                          data_shape_lens.begin() + batch_dims,
                                                          1,
                                                          std::multiplies<std::size_t>());
                std::size_t data_batch_stride =
                    std::accumulate(data_shape_lens.begin() + batch_dims,
                                    data_shape_lens.end(),
                                    1,
                                    std::multiplies<std::size_t>());
                auto num_slices_per_batch = num_slices / num_batches;

                std::vector<std::size_t> sizes_from_slice_dims(num_slice_dims);
                {
                    auto running_product = slice_size;
                    for(std::size_t i = 0; i < num_slice_dims; ++i)
                    {
                        sizes_from_slice_dims[num_slice_dims - 1 - i] = running_product;
                        running_product *= data_shape_lens[batch_dims + num_slice_dims - 1 - i];
                    }
                }

                std::vector<std::size_t> input_slice_offsets(num_slices);
                par_for(num_slices, [&](const auto i) {
                    std::size_t batch_idx = i / num_slices_per_batch;

                    auto slice_indices                = indices.begin() + (i * num_slice_dims);
                    std::size_t relative_slice_offset = 0;
                    for(size_t dim_idx = 0; dim_idx < num_slice_dims; ++dim_idx)
                    {
                        int64_t index                   = *(slice_indices + dim_idx);
                        const std::size_t input_dim_idx = batch_dims + dim_idx;
                        const auto input_dim            = data_shape_lens[input_dim_idx];
                        if(index < -static_cast<int64_t>(input_dim) or
                           index >= static_cast<int64_t>(input_dim))
                            MIGRAPHX_THROW("GatherND: index " + std::to_string(index) +
                                           " is out of bounds for dim of len " +
                                           std::to_string(input_dim));
                        if(index < 0)
                            index += input_dim;

                        relative_slice_offset += index * sizes_from_slice_dims[dim_idx];
                    }

                    input_slice_offsets[i] =
                        (batch_idx * data_batch_stride) + relative_slice_offset;
                });

                par_for(num_slices * slice_size, [&](const auto i) {
                    auto slice_offset = input_slice_offsets[i / slice_size];
                    output[i]         = data[slice_offset + i % slice_size];
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
