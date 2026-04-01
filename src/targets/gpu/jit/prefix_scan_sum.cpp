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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static const char* const prefix_scan_sum_kernel = R"__migraphx__(
#include <migraphx/kernels/prefix_scan_sum.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void prefix_scan_sum_kernel(void* input_p, void* output_p)
{
    make_tensors()(input_p, output_p)([](auto input, auto output) {
        auto idx = make_index();
        const index_int nslices = ${nslices};
        const index_int n = ${n};
        const index_int axis_stride = ${axis_stride};
        index_int slice_idx = idx.group;
        if(slice_idx < nslices)
        {
            index_int offset = ${offset_computation};
            prefix_scan_sum_slice<${block_size}, ${exclusive}, ${reverse}>(
                input, output, offset, n, axis_stride);
        }
    });
}

}

} // namespace migraphx

)__migraphx__";

struct prefix_scan_sum_compiler : compiler<prefix_scan_sum_compiler>
{
    std::vector<std::string> names() const { return {"prefix_scan_sum"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = inputs;
        options.kernel_name    = "prefix_scan_sum_kernel";

        const auto& output_shape = inputs.back();
        auto exclusive           = v.get("exclusive", false);
        auto reverse             = v.get("reverse", false);

        std::size_t axis = v.at("axis").to<std::size_t>();

        std::size_t n           = output_shape.lens()[axis];
        std::size_t axis_stride = output_shape.strides()[axis];

        std::size_t nslices = 1;
        for(std::size_t i = 0; i < output_shape.lens().size(); ++i)
        {
            if(i != axis)
                nslices *= output_shape.lens()[i];
        }

        auto ndim           = output_shape.lens().size();
        const auto& lens    = output_shape.lens();
        const auto& strides = output_shape.strides();

        std::vector<std::size_t> batch_lens;
        std::vector<std::size_t> batch_strides;
        for(std::size_t i = 0; i < ndim; ++i)
        {
            if(i != axis)
            {
                batch_lens.push_back(lens[i]);
                batch_strides.push_back(strides[i]);
            }
        }

        std::string offset_computation = "0";
        if(not batch_lens.empty())
        {
            std::vector<std::size_t> divisors(batch_lens.size());
            divisors.back() = 1;
            for(std::size_t i = batch_lens.size() - 1; i > 0; --i)
            {
                divisors[i - 1] = divisors[i] * batch_lens[i];
            }

            std::vector<std::string> terms;
            for(std::size_t i = 0; i < batch_lens.size(); ++i)
            {
                std::string idx_expr;
                if(divisors[i] == 1)
                {
                    idx_expr = "slice_idx";
                }
                else
                {
                    idx_expr = "(slice_idx / ";
                    idx_expr += std::to_string(divisors[i]);
                    idx_expr += ")";
                }

                if(i > 0)
                {
                    std::string wrapped = "(";
                    wrapped += idx_expr;
                    wrapped += " % ";
                    wrapped += std::to_string(batch_lens[i]);
                    wrapped += ")";
                    idx_expr = std::move(wrapped);
                }

                if(batch_strides[i] != 0)
                {
                    if(batch_strides[i] == 1)
                    {
                        terms.push_back(idx_expr);
                    }
                    else
                    {
                        idx_expr += " * ";
                        idx_expr += std::to_string(batch_strides[i]);
                        terms.push_back(idx_expr);
                    }
                }
            }

            if(not terms.empty())
            {
                offset_computation = terms[0];
                for(std::size_t i = 1; i < terms.size(); ++i)
                {
                    offset_computation += " + ";
                    offset_computation += terms[i];
                }
            }
        }

        constexpr std::size_t block_size = 256;
        options.global                   = nslices * block_size;
        options.local                    = block_size;

        auto src = interpolate_string(prefix_scan_sum_kernel,
                                      {{"block_size", std::to_string(block_size)},
                                       {"n", std::to_string(n)},
                                       {"axis_stride", std::to_string(axis_stride)},
                                       {"nslices", std::to_string(nslices)},
                                       {"offset_computation", offset_computation},
                                       {"exclusive", exclusive ? "true" : "false"},
                                       {"reverse", reverse ? "true" : "false"}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
