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
#include <migraphx/par_dfor.hpp>

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
        return pack(f(self.has_ignore_index, "has_ignore_index"),
                    f(self.weighted, "weighted"),
                    f(self.mode, "mode"));
    }

    value attributes() const { return {{"has_ignore_index", "weighted", "mode"}}; }

    std::string name() const { return "crossentropyloss"; }

    migraphx::shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(4);

        auto log_scores   = inputs[0];
        auto label        = inputs[1];
        auto weights      = inputs[2];
        auto ignore_index = inputs[3];

        if(not ignore_index.scalar())
        {
            MIGRAPHX_THROW("crossentropyloss: Ignore index must be scalar shape");
        }

        if(log_scores.lens()[0] != label.lens()[0])
        {
            MIGRAPHX_THROW("crossentropyloss: Score and Labels must identical batch size inputs");
        }

        if(log_scores.ndim() <= label.ndim())
        {
            MIGRAPHX_THROW(
                "crossentropyloss: Score and Labels must contain identical K-Dimensions");
        }

        if(weights.lens()[0] != log_scores.lens()[1])
        {
            MIGRAPHX_THROW(
                "Invalid weight vector shape. Weight must contain weight for each class");
        }

        // Output of loss tensor and rearranged weights should have the same shape as input labels
        std::vector<migraphx::shape> output_shapes{label, weights};

        return shape{output_shapes};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape.sub_shapes().at(0)};
        argument weight_vec{output_shape.sub_shapes().at(1)};

        auto log_scores_input = args.at(0);
        auto labels           = args.at(1);
        auto weights          = args.at(2);
        auto ignore_index     = args.at(3);

        auto log_scores_shape = log_scores_input.get_shape();
        auto label_shape      = labels.get_shape();
        auto weight_shape     = weights.get_shape();

        // int batch_size  = log_scores_shape.lens().at(0);
        int num_classes = log_scores_shape.lens().at(1);

        visit_all(result, weight_vec, log_scores_input, labels, weights)(
            [&](auto output, auto weight, auto data, auto label, auto in_weight) {
                par_dfor(num_classes)([&](int class) {
                    int c = label[class];

                    weight[c] = in_weight[c];
                    output[c] = 0;

                    if(self.has_ignore_index && c != ignore_index)
                    {
                        output[c] = data[class];
                        if(self.weighted)
                        {
                            output[c] *= weight[c];
                        }
                    }
                });
            });

        std::vector<argument> output_vec{result, weight_vec};
        argument output_result;
        return output_result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
