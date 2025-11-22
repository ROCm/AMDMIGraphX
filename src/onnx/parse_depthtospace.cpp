/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_depthtospace : op_parser<parse_depthtospace>
{
    std::vector<op_desc> operators() const { return {{"DepthToSpace"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto s = args[0]->get_shape();
        // mode attribute of DepthToSpace
        auto mode = std::string("DCR");
        if(contains(info.attributes, "mode"))
        {
            mode = info.attributes.at("mode").s(); // DCR or CRD?
        }
        // blocksize attribute of DepthToSpace
        int64_t blocksize = 0;
        if(contains(info.attributes, "blocksize"))
        {
            blocksize = static_cast<int64_t>(info.attributes.at("blocksize").i());
        }
        if(blocksize < 1)
        {
            MIGRAPHX_THROW("DepthToSpace: blocksize is less than 1");
        }
        // In the dynamic case, we need to compute the n, h, and w values at runtime.
        // However, currently we don't have a way to represent them symbolically.
        // Instead, we can use the "dimensions_of" op to represent the actual n,h,w values
        // at runtime and perform the corresponding arithmetic (multiplying or dividing by
        // blocksize).
        if(s.dynamic())
        {
            auto dyn_dims1              = s.dyn_dims(); // the expanded dims vector (6d)
            auto dyn_dims2              = s.dyn_dims(); // the final transformed vector (4d)
            int64_t divisor             = blocksize * blocksize;
            uint64_t blocksize_unsigned = blocksize;
            auto blocksize_literal      = info.add_literal({blocksize});

            auto n = info.add_instruction(make_op("dimensions_of", {{"end", 1}}), args[0]);
            if(not dyn_dims1[1].is_fixed())
            {
                MIGRAPHX_THROW("DepthToSpace: dynamic channels are not supported");
            }
            int64_t c = dyn_dims1[1].max;
            auto h =
                info.add_instruction(make_op("dimensions_of", {{"start", 2}, {"end", 3}}), args[0]);
            auto w =
                info.add_instruction(make_op("dimensions_of", {{"start", 3}, {"end", 4}}), args[0]);

            auto c_div = info.add_literal({c / divisor});

            dyn_dims2[1] = {dyn_dims2[1].min / divisor, dyn_dims2[1].max / divisor};
            dyn_dims2[2] = dyn_dims2[2] * blocksize_unsigned;
            dyn_dims2[3] = dyn_dims2[3] * blocksize_unsigned;
            // push back h and w to expand the vector to 6d
            dyn_dims1.push_back(dyn_dims1[2]);
            dyn_dims1.push_back(dyn_dims1[3]);
            dyn_dims1[2] = {blocksize_unsigned, blocksize_unsigned, {}};

            std::vector<int64_t> perm;
            instruction_ref new_shape1;
            instruction_ref new_shape_alloc1;
            if(mode == "DCR")
            {
                // expanded vector = n, blocksize, blocksize, c // (blocksize**2), h, w
                dyn_dims1[3] = {dyn_dims1[1].min / divisor, dyn_dims1[1].max / divisor, {}};
                dyn_dims1[1] = {blocksize_unsigned, blocksize_unsigned, {}};
                perm         = {0, 3, 4, 1, 5, 2};
                new_shape1   = info.add_instruction(
                    make_op("concat"), n, blocksize_literal, blocksize_literal, c_div, h, w);
                new_shape_alloc1 = info.add_instruction(
                    make_op("allocate",
                            {{"shape", to_value(migraphx::shape{s.type(), dyn_dims1})}}),
                    new_shape1);
                auto reshape1 = info.add_instruction(make_op("reshape"), args[0], new_shape_alloc1);
                auto transpose =
                    info.add_instruction(make_op("transpose", {{"permutation", perm}}), reshape1);
                auto h_blocksize = info.add_instruction(make_op("mul"), h, blocksize_literal);
                auto w_blocksize = info.add_instruction(make_op("mul"), w, blocksize_literal);
                auto new_shape2 =
                    info.add_instruction(make_op("concat"), n, c_div, h_blocksize, w_blocksize);
                auto new_shape_alloc2 = info.add_instruction(
                    make_op("allocate",
                            {{"shape", to_value(migraphx::shape{s.type(), dyn_dims2})}}),
                    new_shape2);
                return info.add_instruction(make_op("reshape"), transpose, new_shape_alloc2);
            }
            else if(mode == "CRD")
            {
                // expanded vector = b, c // (blocksize ** 2), blocksize, blocksize, h, w
                dyn_dims1[1] = {dyn_dims1[1].min / divisor, dyn_dims1[1].max / divisor, {}};
                dyn_dims1[3] = {blocksize_unsigned, blocksize_unsigned, {}};
                perm         = {0, 1, 4, 2, 5, 3};
                new_shape1   = info.add_instruction(
                    make_op("concat"), n, c_div, blocksize_literal, blocksize_literal, h, w);
                new_shape_alloc1 = info.add_instruction(
                    make_op("allocate",
                            {{"shape", to_value(migraphx::shape{s.type(), dyn_dims1})}}),
                    new_shape1);
                auto reshape1 = info.add_instruction(make_op("reshape"), args[0], new_shape_alloc1);
                auto transpose =
                    info.add_instruction(make_op("transpose", {{"permutation", perm}}), reshape1);
                auto h_blocksize = info.add_instruction(make_op("mul"), h, blocksize_literal);
                auto w_blocksize = info.add_instruction(make_op("mul"), w, blocksize_literal);
                auto new_shape2 =
                    info.add_instruction(make_op("concat"), n, c_div, h_blocksize, w_blocksize);
                auto new_shape_alloc2 = info.add_instruction(
                    make_op("allocate",
                            {{"shape", to_value(migraphx::shape{s.type(), dyn_dims2})}}),
                    new_shape2);
                return info.add_instruction(make_op("reshape"), transpose, new_shape_alloc2);
            }
        }
        // calculate dimensions
        auto lens1            = s.lens();
        auto lens2            = s.lens();
        unsigned long divisor = std::pow(blocksize, 2);
        if((lens2[1] % divisor) == 0)
            lens2[1] = lens2[1] / divisor;
        else
            MIGRAPHX_THROW("DepthToSpace: div by blocksize quotient not int");
        lens1.push_back(lens1[2]);
        lens1.push_back(lens1[3]);
        lens2[2] = lens2[2] * blocksize;
        lens2[3] = lens2[3] * blocksize;
        lens1[2] = blocksize;
        std::vector<int64_t> perm;
        if(mode == "DCR")
        {
            lens1[3] = lens1[1] / divisor;
            lens1[1] = blocksize;
            perm     = {0, 3, 4, 1, 5, 2};
        }
        else if(mode == "CRD")
        {
            lens1[1] = lens1[1] / divisor;
            lens1[3] = blocksize;
            perm     = {0, 1, 4, 2, 5, 3};
        }
        else
            MIGRAPHX_THROW("DepthToSpace: mode attribute cannot be read.");

        auto temp1 = info.add_instruction(make_op("reshape", {{"dims", lens1}}), args[0]);
        auto temp2 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), temp1);
        return info.add_instruction(make_op("reshape", {{"dims", lens2}}), temp2);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
