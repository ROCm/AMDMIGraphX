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
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/simplify_qlinear_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct eliminate_zero_point
{
    auto get_qlinear_ops_names() const
    {
        static std::unordered_set<std::string> qdq_names = {"quantizelinear", "dequantizelinear"};
        return qdq_names;
    }
    auto matcher() const
    {
        return match::name(get_qlinear_ops_names())(
            match::arg(0)(match::any().bind("x")),
            match::arg(1)(match::any().bind("scale")),
            match::arg(2)(match::has_value(0.0f, 0, 0).bind("zp")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x     = r.instructions["x"];
        auto scale = r.instructions["scale"];

        auto op = ins->get_operator().to_value();
        if(ins->get_operator().name() == "quantizelinear")
        {
            op["out_type"] = to_value(ins->get_shape().type());
        }
        auto qdq_ins = m.insert_instruction(ins, migraphx::make_op(ins->name(), op), {x, scale});
        m.replace_instruction(ins, qdq_ins);
    }
};

void simplify_qlinear_ops::apply(module& m) const
{
    match::find_matches(m, eliminate_zero_point{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
