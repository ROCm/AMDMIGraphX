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
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pmr/vector.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(code_object_op);

shape code_object_op::compute_shape(std::vector<shape> inputs) const
{
    std::transform(inputs.begin(), inputs.end(), inputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    auto einputs = expected_inputs;
    std::transform(einputs.begin(), einputs.end(), einputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    if(not migraphx::equal(flatten(einputs), flatten(inputs), &shape::is_compatible))
        MIGRAPHX_THROW("Input shapes have changed: [" + to_string_range(einputs) + "] -> [" +
                       to_string_range(inputs) + "]");
    auto output_buffer_shape = inputs.at(get_output_arg(inputs.size()));
    if(not shape::is_compatible(output_buffer_shape, output))
        MIGRAPHX_THROW("Output buffer [" + to_string(output_buffer_shape) +
                       "] doesn't match the expected output shape from the kernel [" +
                       to_string(output) + "]");
    return output;
}

static bool needs_flatten(const std::vector<argument>& args)
{
    return std::any_of(args.begin(), args.end(), [&](const argument& arg) {
        return arg.get_shape().type() == shape::tuple_type;
    });
}

template <class F>
static void visit_flatten_args(const std::vector<argument>& args, F f)
{
    if(needs_flatten(args))
        f(flatten(args));
    else
        f(args);
}

argument
code_object_op::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
#if MIGRAPHX_HAS_PMR
    std::array<char, 256> storage;
    std::pmr::monotonic_buffer_resource resource{storage.data(), storage.size()};
    pmr::vector<void*> kargs(&resource);
#else
    pmr::vector<void*> kargs;
#endif
    visit_flatten_args(args, [&](const auto& fargs) {
        kargs.reserve(fargs.size());
        std::transform(fargs.begin(),
                       fargs.end(),
                       std::back_inserter(kargs),
                       [](const argument& a) { return a.data(); });
    });
    auto [start, stop] = ctx.get_perf_events();


    if(kernel_args.count("op") > 0 &&
       kernel_args.at("op").to<int>() == static_cast<int>(mlss_op_type::mha))
    {
        auto query = args[0];

        auto query_shape   = query.get_shape().lens();
        int head_dim     = query_shape[3];

        auto query_strides = query.get_shape().strides();                

        int batch_size         = query_shape[0];
        int sequence_length  = query_shape[2];
        int head_num           = query_shape[1];

        auto outval              = args[4];
        auto outval_strides     = outval.get_shape().strides();


        std::vector<kernel_argument> kargs_input;

        // kernel_argument stores a raw pointer so requires a new named variable 
        hipDeviceptr_t d_q_in    = query.data();
        kargs_input.emplace_back(d_q_in);

        hipDeviceptr_t d_out_in    = outval.data();
        kargs_input.emplace_back(d_out_in);      

        kargs_input.push_back(batch_size);
        kargs_input.push_back(sequence_length);
        kargs_input.push_back(head_num);
        kargs_input.push_back(head_dim);

        float scale_ka = kernel_args.at("scale").to<float>();
        kargs_input.push_back(scale_ka);

        // -----------------------------------------------------------------------
        // Strides for the [B, S, H, 3*D] QKV layout (seq-major, no transpose needed):
        //   d0 = S * H * 3*D   (batch stride — same total as head-major since B=1)
        //   d1 = 3*D           (head stride — heads are innermost, so stride is just 3*D)
        //   d2 = H * 3*D       (sequence stride — each seq step skips H * 3*D elements)
        //   d3 = 1             (element stride)
        // Output has standard [B, H, S, D] layout (no interleaving).
        // -----------------------------------------------------------------------
        uint32_t stride_d0 = static_cast<uint32_t>(sequence_length * head_num * 3 * head_dim);
        uint32_t stride_d1 = static_cast<uint32_t>(3 * head_dim);
        uint32_t stride_d2 = static_cast<uint32_t>(head_num * 3 * head_dim);
        uint32_t stride_d3 = 1u;

        // q
        uint32_t q_stride_d0 = stride_d0;
        uint32_t q_stride_d1 = stride_d1;
        uint32_t q_stride_d2 = stride_d2;
        uint32_t q_stride_d3 = stride_d3;

        // k
        uint32_t k_stride_d0 = stride_d0;
        uint32_t k_stride_d1 = stride_d1;
        uint32_t k_stride_d2 = stride_d2;
        uint32_t k_stride_d3 = stride_d3;

        // v
        uint32_t v_stride_d0 = stride_d0;
        uint32_t v_stride_d1 = stride_d1;
        uint32_t v_stride_d2 = stride_d3; // swapped for v
        uint32_t v_stride_d3 = stride_d2;

        kargs_input.push_back(q_stride_d0);
        kargs_input.push_back(q_stride_d1);
        kargs_input.push_back(q_stride_d2);
        kargs_input.push_back(q_stride_d3);

        kargs_input.push_back(k_stride_d0);
        kargs_input.push_back(k_stride_d1);
        kargs_input.push_back(k_stride_d2);
        kargs_input.push_back(k_stride_d3);

        kargs_input.push_back(v_stride_d0);
        kargs_input.push_back(v_stride_d1);
        kargs_input.push_back(v_stride_d2);
        kargs_input.push_back(v_stride_d3);

        // output strides
        uint32_t output_stride_d0 = outval_strides[0];
        uint32_t output_stride_d1 = outval_strides[1];
        uint32_t output_stride_d2 = outval_strides[2];
        uint32_t output_stride_d3 = outval_strides[3];

        kargs_input.push_back(output_stride_d0);
        kargs_input.push_back(output_stride_d1);
        kargs_input.push_back(output_stride_d2);
        kargs_input.push_back(output_stride_d3);

        const int grid  = batch_size * head_num * sequence_length * 2;
        const int block = 128;

        k.launch(ctx.get_stream().get(), grid, block, kargs_input, start, stop);
        return args[4];
    }
    else
    {
        k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);
        return args[get_output_arg(args.size())];
    }
}
void code_object_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
