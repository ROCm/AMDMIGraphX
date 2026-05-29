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
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/op/insert_slice.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

static const char* const insert_slice_static_kernel = R"__migraphx__(
#include <migraphx/kernels/insert_slice.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/array.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {
MIGRAPHX_GLOBAL void insert_slice_kernel(void* input0_p, void* input1_p)
{
    auto idx     = make_index();
    auto offsets = index_ints<${offsets}>{};
    auto strides = index_ints<${strides}>{};
    make_tensors()(input0_p, input1_p)([&](auto source, auto output) {
        insert_slice<_rank_, index_ints<${offsets}>, index_ints<${strides}>, _deref_dest_>(
            idx, offsets, strides, source, output);
    });
}
}

} // namespace migraphx

)__migraphx__";

// Kernel for dynamic offsets: (source, dest, offsets_tensor); writes in-place to dest.
static const char* const insert_slice_dynamic_offsets_kernel = R"__migraphx__(
#include <migraphx/kernels/insert_slice.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/array.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {
MIGRAPHX_GLOBAL void insert_slice_kernel(void* input0_p, void* input1_p, void* input2_p)
{
    auto idx     = make_index();
    auto strides = index_ints<${strides}>{};
    make_tensors()(input0_p, input1_p, input2_p)([&](auto source, auto output, auto offsets_tensor) {
        insert_slice<_rank_, _batched_offsets_, index_ints<${strides}>, _deref_dest_>(
            idx, offsets_tensor, strides, source, output);
    });
}
}

} // namespace migraphx

)__migraphx__";

struct insert_slice_compiler : compiler<insert_slice_compiler>
{
    std::vector<std::string> names() const { return {"insert_slice"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // After lowering: 2 args = (source, dest), 3 args = (source, dest, offsets_tensor)
        if(inputs.size() < 2)
            MIGRAPHX_THROW("insert_slice: expected at least 2 inputs (source, dest)");
        if(inputs.size() > 3)
            MIGRAPHX_THROW("insert_slice: dynamic sizes/strides as inputs not yet supported on GPU");

        auto rank = inputs[0].ndim();
        std::vector<std::size_t> static_offsets;
        std::vector<std::size_t> static_strides;
        if(v.contains("static_offsets"))
            static_offsets = v.at("static_offsets").to_vector<std::size_t>();
        if(v.contains("static_strides"))
            static_strides = v.at("static_strides").to_vector<std::size_t>();
        bool deref_dest = v.get("deref_dest", false);

        if(static_offsets.size() < rank)
            static_offsets.resize(rank, 0);
        if(static_strides.size() < rank)
            static_strides.resize(rank, 1);

        hip_compile_options options;
        const auto& out_s = inputs[1]; // in-place output = destination tensor
        const auto& in_s = inputs[0]; // source tensor
        options.set_launch_params(v, compute_global_for(ctx, in_s.elements()));
        options.inputs      = inputs;
        options.output      = out_s;
        options.kernel_name = "insert_slice_kernel";
        // In-place into dest (input 1), including when a trailing offsets tensor is present
        options.output_arg = 1;

        std::string strides_str = to_string_range(static_strides);

        if(inputs.size() == 2)
        {
            // Static offsets: (source, dest)
            std::string offsets_str = to_string_range(static_offsets);
            std::string src         = interpolate_string(
                insert_slice_static_kernel,
                {{"rank", std::to_string(rank)},
                 {"offsets", offsets_str},
                 {"strides", strides_str},
                 {"deref_dest", deref_dest ? "true" : "false"}});
            replace_string_inplace(src, "_rank_", std::to_string(rank));
            replace_string_inplace(src, "_deref_dest_", deref_dest ? "true" : "false");
            return compile_hip_code_object(ctx, src, options);
        }

        // inputs.size() == 3: source, dest, offsets (1D [rank] or 2D [batch, rank])
        const bool batched_offsets = inputs[2].ndim() == 2;
        std::string src            = interpolate_string(
            insert_slice_dynamic_offsets_kernel,
            {{"rank", std::to_string(rank)},
             {"strides", strides_str},
             {"deref_dest", deref_dest ? "true" : "false"}});
        replace_string_inplace(src, "_rank_", std::to_string(rank));
        replace_string_inplace(src, "_deref_dest_", deref_dest ? "true" : "false");
        replace_string_inplace(src, "_batched_offsets_", batched_offsets ? "true" : "false");
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
