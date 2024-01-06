/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OP_POINTWISE_HPP
#define MIGRAPHX_GUARD_OP_POINTWISE_HPP

#include <migraphx/config.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/module.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pointwise
{
    std::string name() const { return "pointwise"; }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
        {
            MIGRAPHX_THROW("should have one submodule.");
        }
        auto* pm = mods.front();
        if(pm->get_output_shapes().size() != 1)
            MIGRAPHX_THROW("pointwise should have only one output.");
        if(inputs.empty())
            MIGRAPHX_THROW("pointwise should have at least one input");
        auto pnames = pm->get_parameter_names();
        std::sort(pnames.begin(), pnames.end());
        check_shapes{inputs, *this}.has(pnames.size()).same_dims();

        auto type = pm->get_output_shapes().front().type();

        // Scalar output if all inputs are scalar
        if(inputs.front().elements() == 1 and
           all_of(inputs, [](const auto& s) { return s.scalar(); }))
            return shape{type};

        return shape::from_permutation(type, inputs.front().lens(), find_permutation(inputs));
    }

    argument compute(const shape& output_shape,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        argument output{output_shape};
        auto* pm    = mods.front();
        auto pnames = pm->get_parameter_names();
        std::sort(pnames.begin(), pnames.end());

        par_for(output_shape.elements(), [&](auto i) {
            std::unordered_map<std::string, argument> params;

            std::transform(
                pnames.begin(),
                pnames.end(),
                args.begin(),
                std::inserter(params, params.end()),
                [&](auto&& name, auto&& arg) { return std::make_pair(name, arg.element(i)); });

            auto results = run(pm, params);
            assert(results.size() == 1);
            visit_all(output, results.front())([&](auto out, auto x) { out[i] = x.front(); });
        });
        return output;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_OP_POINTWISE_HPP
