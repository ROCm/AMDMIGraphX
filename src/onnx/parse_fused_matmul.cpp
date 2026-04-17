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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/op/builder/insert.hpp>

#include <algorithm>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// com.microsoft.FusedMatMul
// Matrix product that behaves like numpy.matmul with optional transposes and a scalar alpha.
//
// Attributes:
//   alpha       : float, scalar multiplier applied to the product (default 1.0)
//   transA      : int,   transpose A on the last two dims before the multiply (default 0)
//   transB      : int,   transpose B on the last two dims before the multiply (default 0)
//   transBatchA : int,   for rank-R A, permute [1, 2, ..., R-2, 0, R-1] (default 0)
//   transBatchB : int,   for rank-R B, permute [1, 2, ..., R-2, 0, R-1] (default 0)
//
// transBatch* is applied before the corresponding trans*; if either transBatch is set,
// both inputs must have the same rank and rank >= 3 (matches ORT's MatMulComputeHelper).

struct parse_fused_matmul : op_parser<parse_fused_matmul>
{
    std::vector<op_desc> operators() const { return {{"FusedMatMul"}}; }

    static instruction_ref apply_trans_batch(const onnx_parser::node_info& info,
                                             instruction_ref x)
    {
        auto r = x->get_shape().ndim();
        std::vector<int64_t> perm(r);
        std::iota(perm.begin(), perm.end(), 0);
        // [0, 1, 2, ..., R-1] -> [1, 2, ..., R-2, 0, R-1]
        std::rotate(perm.begin(), perm.begin() + 1, perm.begin() + (r - 1));
        return info.add_instruction(make_op("transpose", {{"permutation", perm}}), x);
    }

    static instruction_ref apply_trans_last_two(const onnx_parser::node_info& info,
                                                instruction_ref x)
    {
        auto r = x->get_shape().ndim();
        std::vector<int64_t> perm(r);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[r - 2], perm[r - 1]);
        return info.add_instruction(make_op("transpose", {{"permutation", perm}}), x);
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(args.size() != 2)
        {
            MIGRAPHX_THROW("PARSE_FUSEDMATMUL: expected 2 inputs, got " +
                           std::to_string(args.size()));
        }

        float alpha        = 1.0f;
        bool trans_a       = false;
        bool trans_b       = false;
        bool trans_batch_a = false;
        bool trans_batch_b = false;

        if(contains(info.attributes, "alpha"))
        {
            alpha = parser.parse_value(info.attributes.at("alpha")).at<float>();
        }
        if(contains(info.attributes, "transA"))
        {
            trans_a = parser.parse_value(info.attributes.at("transA")).at<bool>();
        }
        if(contains(info.attributes, "transB"))
        {
            trans_b = parser.parse_value(info.attributes.at("transB")).at<bool>();
        }
        if(contains(info.attributes, "transBatchA"))
        {
            trans_batch_a = parser.parse_value(info.attributes.at("transBatchA")).at<bool>();
        }
        if(contains(info.attributes, "transBatchB"))
        {
            trans_batch_b = parser.parse_value(info.attributes.at("transBatchB")).at<bool>();
        }

        auto a0 = args[0];
        auto a1 = args[1];
        auto s0 = a0->get_shape();
        auto s1 = a1->get_shape();

        if(s0.dynamic() or s1.dynamic())
        {
            MIGRAPHX_THROW("PARSE_FUSEDMATMUL: dynamic inputs not supported");
        }

        const auto r0 = s0.ndim();
        const auto r1 = s1.ndim();

        if(trans_batch_a or trans_batch_b)
        {
            if(r0 != r1 or r0 < 3)
            {
                MIGRAPHX_THROW("PARSE_FUSEDMATMUL: transBatchA/transBatchB require both inputs to "
                               "have the same rank >= 3");
            }
        }

        // numpy.matmul 1-D promotion: only applied when no transBatch is requested. With
        // transBatch set, rank must already be >= 3 (enforced above).
        bool is_a_prepended = false;
        bool is_b_appended  = false;
        if(r0 == 1)
        {
            is_a_prepended = true;
            a0 = op::builder::add("unsqueeze", *info.mod, {a0}, {{"axes", {0}}}).at(0);
        }
        if(r1 == 1)
        {
            is_b_appended = true;
            a1 = op::builder::add("unsqueeze", *info.mod, {a1}, {{"axes", {1}}}).at(0);
        }

        // transBatch* is applied before trans*, matching ORT's MatMulComputeHelper.
        if(trans_batch_a)
            a0 = apply_trans_batch(info, a0);
        if(trans_batch_b)
            a1 = apply_trans_batch(info, a1);
        if(trans_a)
            a0 = apply_trans_last_two(info, a0);
        if(trans_b)
            a1 = apply_trans_last_two(info, a1);

        auto res = op::builder::add("dot", *info.mod, {a0, a1}).at(0);

        if(not float_equal(alpha, 1.0f))
        {
            auto dtype     = res->get_shape().type();
            auto alpha_lit = info.add_literal(literal{shape{dtype}, {alpha}});
            res            = info.add_common_op("mul", res, alpha_lit);
        }

        // Undo 1-D promotion by squeezing the prepended/appended axes.
        int64_t num_axis = res->get_shape().ndim();
        if(is_a_prepended)
        {
            res = op::builder::add(
                      "squeeze", *info.mod, {res}, {{"axes", {num_axis - 2}}})
                      .at(0);
            --num_axis;
        }
        if(is_b_appended)
        {
            res = op::builder::add(
                      "squeeze", *info.mod, {res}, {{"axes", {num_axis - 1}}})
                      .at(0);
        }

        return res;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
