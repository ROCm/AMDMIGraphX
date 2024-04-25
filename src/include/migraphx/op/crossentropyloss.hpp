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
#ifndef MIGRAPHX_GUARD_OPERATORS_CROSSENTROPYLOSS_HPP
#define MIGRAPHX_GUARD_OPERATORS_CROSSENTROPYLOSS_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct crossentropyloss
{
    bool has_ignore_index = false;
    bool weighted         = false;
    std::string mode      = "mean";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.ignore_index, "has_ignore_index"),
                    f(self.weighted, "weighted"),
                    f(self.mode, "mode"));
    }

    std::string name() const { return "crossentropyloss"; }

    value attributes() const { return {{"has_ignore_index", "weighted", "mode"}}; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(4);

        auto log_scores_input = inputs.at(0);
        auto labels           = inputs.at(1);

        auto log_scores_shape = log_scores_input->get_shape();
        auto label_shape      = labels->get_shape();

        if(log_scores_shape.lens()[0] != label_shape.lens()[0])
        {
            MIGRAPHX_THROW("crossentropyloss: Score and Labels must identical batch size inputs");
        }

        if(log_scores_shape.ndims() - 1 != label_shape.ndims())
        {
            MIGRAPHX_THROW(
                "crossentropyloss: Score and Labels must contain identical K-Dimensions");
        }

        auto weights      = inputs.at(2);
        auto ignore_index = inputs.at(3);
        check_shape{ignore_index.get_shape()}.scalar();

        // Need to compute additional output for W vector for weighted mean reduction
        if(self.weighted and self.mode == "mean") {}

        // Output of loss tensor should have the same shape as input labels, output weight vector as
        // well
        std::vector<size_t> output_lens{log_scores_input->get_shape().lens()};
        if(self.mode)

            return output_lens;
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {

        int batch_size  = log_scores_shape.lens()[0];
        int num_classes = log_scores_shape.lens()[1];

        argument result { output_shape }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
