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
#ifndef MIGRAPHX_GUARD_OPERATORS_DECONVOLUTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_DECONVOLUTION_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/par_dfor.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct deconvolution
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

    std::string name() const { return "deconvolution"; }

    void check_attribute_size() const
    {
        if(not((padding.size() == stride.size() or (padding.size() / 2) == stride.size()) and
               stride.size() == dilation.size()))
        {
            MIGRAPHX_THROW("deconvolution: inconsistent attribute sizes");
        }
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().min_ndims(3);

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        size_t kdims         = input.lens().size() - 2;
        if(kdims != this->kdims())
        {
            MIGRAPHX_THROW("deconvolution: input k-dims does not match attribute size");
        }

        std::vector<size_t> output_lens{input.lens()[0], weights.lens()[1]};

        for(size_t i = 0; i < kdims; i++)
        {
            output_lens.push_back(std::size_t(std::max<std::ptrdiff_t>(
                1,
                stride[i] * (input.lens()[i + 2] - 1) +
                    ((weights.lens()[i + 2] - 1) * dilation[i] + 1) - 2 * padding[i])));
        }
        return inputs[0].with_lens(output_lens);
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto kdims = this->kdims();
        visit_all(result, args[0], args[1])([&](auto output, auto input, auto weights) {
            using type = typename decltype(output)::value_type;

            std::fill(output.begin(), output.end(), type{0});

            auto in_lens = input.get_shape().lens();
            auto in_n    = in_lens[0];
            auto in_c    = in_lens[1];

            auto wei   = weights.get_shape().lens();
            auto wei_n = wei[0];
            auto wei_c = wei[1];

            auto out_lens = output_shape.lens();

            std::vector<std::size_t> win_size{in_c};
            std::copy(in_lens.begin() + 2, in_lens.end(), std::back_inserter(win_size));
            std::copy(wei.begin() + 2, wei.end(), std::back_inserter(win_size));
            shape win_shape{output_shape.type(), win_size};

            par_dfor(in_n, wei_c)([&](int o, int k) {
                shape_for_each(win_shape, [&](auto idx_win) {
                    const int w = idx_win[0];

                    auto input_dims_start = idx_win.begin() + 1;
                    auto wei_dims_start   = idx_win.begin() + kdims + 1;

                    std::vector<std::ptrdiff_t> win_start;
                    for(std::size_t n = 0; n < kdims; ++n)
                    {
                        win_start.push_back(std::ptrdiff_t(*(input_dims_start + n) * stride[n]) -
                                            std::ptrdiff_t(padding[n]));
                    }

                    const int group_id = w / (wei_n / group);
                    const int in_ch    = group_id * wei_c + k;

                    std::vector<std::ptrdiff_t> idx_out{o, in_ch};

                    for(size_t n = 0; n < kdims; n++)
                    {
                        idx_out.push_back(win_start[n] + *(wei_dims_start + n) * dilation[n]);
                    }

                    std::vector<std::ptrdiff_t> idx_wei{w, k};
                    std::copy(wei_dims_start, idx_win.end(), std::back_inserter(idx_wei));

                    std::vector<std::ptrdiff_t> idx_in{o, w};
                    std::copy(input_dims_start, wei_dims_start, std::back_inserter(idx_in));

                    if(std::all_of(
                           idx_out.begin() + 2, idx_out.end(), [&](auto ii) { return ii >= 0; }) and
                       std::equal(idx_out.begin() + 2,
                                  idx_out.end(),
                                  out_lens.begin() + 2,
                                  out_lens.end(),
                                  std::less<std::ptrdiff_t>{}))
                    {
                        output(idx_out.begin(), idx_out.end()) +=
                            input(idx_in.begin(), idx_in.end()) *
                            weights(idx_wei.begin(), idx_wei.end());
                    }
                });
            });
        });
        return result;
    }

    size_t kdims() const
    {
        check_attribute_size();
        return stride.size();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
