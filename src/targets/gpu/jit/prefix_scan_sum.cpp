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
        const index_int num_batch_dims = ${num_batch_dims};
        const index_int batch_flat_strides[] = {${batch_flat_strides}};
        const index_int batch_tensor_strides[] = {${batch_tensor_strides}};
        index_int slice_idx = idx.group;
        if(slice_idx < nslices)
        {
            index_int offset = 0;
            index_int remaining = slice_idx;
            for(index_int b = 0; b < num_batch_dims; ++b)
            {
                index_int idx_b = remaining / batch_flat_strides[b];
                remaining = remaining % batch_flat_strides[b];
                offset += idx_b * batch_tensor_strides[b];
            }
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

        auto output_shape = inputs.back();
        auto exclusive    = v.get("exclusive", false);
        auto reverse      = v.get("reverse", false);

        int64_t axis_val = v.at("axis").to<int64_t>();
        if(axis_val < 0)
            axis_val += output_shape.ndim();
        std::size_t axis = axis_val;

        std::size_t n           = output_shape.lens()[axis];
        std::size_t axis_stride = output_shape.strides()[axis];

        std::size_t nslices = 1;
        for(std::size_t i = 0; i < output_shape.lens().size(); ++i)
        {
            if(i != axis)
                nslices *= output_shape.lens()[i];
        }

        auto ndim     = output_shape.lens().size();
        auto& lens    = output_shape.lens();
        auto& strides = output_shape.strides();

        // Collect non-axis (batch) dimensions in order, compute flat strides and tensor strides
        std::vector<std::size_t> batch_lens_vec;
        std::vector<std::size_t> batch_tensor_strides_vec;
        for(std::size_t i = 0; i < ndim; ++i)
        {
            if(i != axis)
            {
                batch_lens_vec.push_back(lens[i]);
                batch_tensor_strides_vec.push_back(strides[i]);
            }
        }
        // flat_strides[b] = product of batch_lens[b+1..end], used to unflatten slice_idx.
        // Computed in reverse order (innermost to outermost) so each entry accumulates the
        // sizes of all more-inner batch dimensions.
        std::size_t num_batch = batch_lens_vec.size();
        std::vector<std::size_t> batch_flat_strides_vec(num_batch, 1);
        for(std::size_t b = num_batch; b-- > 0;)
        {
            batch_flat_strides_vec[b] =
                (b + 1 < num_batch) ? batch_flat_strides_vec[b + 1] * batch_lens_vec[b + 1] : 1;
        }

        // Build comma-separated strings; use a dummy "0" for the empty (1-D) case to avoid
        // zero-length arrays in the generated C++ kernel. The loop in the kernel will not
        // execute when num_batch_dims == 0, so the dummy element is never accessed.
        auto to_csv = [](const std::vector<std::size_t>& v) -> std::string {
            if(v.empty())
                return "0";
            std::string s;
            for(std::size_t i = 0; i < v.size(); ++i)
            {
                if(i > 0)
                    s += ", ";
                s += std::to_string(v[i]);
            }
            return s;
        };

        constexpr std::size_t block_size = 256;
        options.global                   = nslices * block_size;
        options.local                    = block_size;

        auto src = interpolate_string(prefix_scan_sum_kernel,
                                      {{"block_size", std::to_string(block_size)},
                                       {"n", std::to_string(n)},
                                       {"axis_stride", std::to_string(axis_stride)},
                                       {"nslices", std::to_string(nslices)},
                                       {"num_batch_dims", std::to_string(num_batch)},
                                       {"batch_flat_strides", to_csv(batch_flat_strides_vec)},
                                       {"batch_tensor_strides", to_csv(batch_tensor_strides_vec)},
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
