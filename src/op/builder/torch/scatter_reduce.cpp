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

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <migraphx/argument.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// scatter_reduce has no native op: use the matching reduction scatter op; for include_self=false
// the target positions are first overwritten with the reduction identity so they drop out.
struct torch_scatter_reduce : op_builder<torch_scatter_reduce>
{
    int64_t dim        = 0;
    std::string reduce = "sum";
    bool include_self  = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.dim, "dim"), f(self.reduce, "reduce"), f(self.include_self, "include_self"));
    }

    static std::vector<std::string> names() { return {"tm::scatter_reduce"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        const std::unordered_map<std::string, std::string> reduce_map = {{"mean", "scatter_none"},
                                                                         {"sum", "scatter_add"},
                                                                         {"prod", "scatter_mul"},
                                                                         {"amax", "scatter_max"},
                                                                         {"amin", "scatter_min"}};
        if(reduce_map.count(reduce) == 0)
            MIGRAPHX_THROW("scatter_reduce: unsupported reduction '" + reduce + "'");

        auto inp  = args[0];
        auto idx  = args[1];
        auto src  = args[2];
        auto axis = tune_axis(inp->get_shape().ndim(), dim, "scatter_reduce");

        if(not include_self and reduce != "mean")
        {
            argument id_arg{shape{inp->get_shape().type(), {1}}};
            id_arg.visit([&](auto v) {
                using type = std::remove_cv_t<typename decltype(v)::value_type>;
                if(reduce == "sum")
                    v.front() = type(0);
                else if(reduce == "prod")
                    v.front() = type(1);
                else if(reduce == "amax")
                    v.front() = std::numeric_limits<type>::lowest();
                else
                    v.front() = std::numeric_limits<type>::max();
            });
            auto identity = m.add_literal(id_arg.get_shape(), id_arg.data());
            identity      = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", idx->get_shape().lens()}}), identity);
            inp = m.insert_instruction(
                ins, make_op("scatter_none", {{"axis", axis}}), {inp, idx, identity});
        }

        return {m.insert_instruction(
            ins, make_op(reduce_map.at(reduce), {{"axis", axis}}), {inp, idx, src})};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
