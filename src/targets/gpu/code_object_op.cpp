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


    if(kernel_args.size() > 0)
    {
        auto query = args[0];

        auto query_shape   = query.get_shape().lens();
        int head_dim     = query_shape[3];

        auto query_strides = query.get_shape().strides();                

        int batch_size         = query_shape[0];
        int q_sequence_length  = query_shape[2];
        int kv_sequence_length = query_shape[2];
        int head_num           = query_shape[1];        

        // std::cout << "Dispatch MHA B: " <<  batch_size << " Seq: " << q_sequence_length << "head_num" << head_num << "head_dim" << head_dim << std::endl;

        int B = batch_size;
        int H = head_num;
        int S = q_sequence_length;  // query sequence length
        int N = kv_sequence_length; // key/value sequence length
        int D = head_dim;           // head dimension

        const int qn = batch_size * head_num * q_sequence_length * head_dim * 3;   

        auto scale              = args[3];        

        auto outval              = args[4];
        auto outval_strides     = outval.get_shape().strides();
        std::size_t outval_bytes = outval.get_shape().bytes();

        std::vector<kernel_argument> kargs;

        hipDeviceptr_t d_q_in    = query.data();
        kargs.emplace_back(d_q_in);

        hipDeviceptr_t d_out_in    = outval.data();
        kargs.emplace_back(d_out_in);      

        kargs.push_back(batch_size);
        kargs.push_back(q_sequence_length);
        kargs.push_back(head_num);
        kargs.push_back(head_dim);

        // hipDeviceptr_t scale_ptr = scale.data();

        // auto scale_elements = scale.get_shape().elements();
        // std::size_t scale_bytes = scale.get_shape().bytes();
        // std::vector<float> scale_out(scale_elements);

        // auto status_scale = hipMemcpy(scale_out.data(), scale_ptr, scale_bytes, hipMemcpyDeviceToHost);
        // if(status_scale != hipSuccess)
        //     MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status_scale));

        //float scale_in              = 0.5f;
        // kargs.push_back(scale_out[0]);

        float scale_ka = kernel_args.at("scale").to<float>();
        kargs.push_back(scale_ka);

        // -----------------------------------------------------------------------
        // Strides for the [B, S, H, 3*D] QKV layout (seq-major, no transpose needed):
        //   d0 = S * H * 3*D   (batch stride — same total as head-major since B=1)
        //   d1 = 3*D           (head stride — heads are innermost, so stride is just 3*D)
        //   d2 = H * 3*D       (sequence stride — each seq step skips H * 3*D elements)
        //   d3 = 1             (element stride)
        // Output has standard [B, H, S, D] layout (no interleaving).
        // -----------------------------------------------------------------------
        uint32_t stride_d0 = static_cast<uint32_t>(q_sequence_length * head_num * 3 * head_dim);
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

        // output strides
        uint32_t output_stride_d0 = outval_strides[0];
        uint32_t output_stride_d1 = outval_strides[1];
        uint32_t output_stride_d2 = outval_strides[2];
        uint32_t output_stride_d3 = outval_strides[3];

        kargs.push_back(q_stride_d0);
        kargs.push_back(q_stride_d1);
        kargs.push_back(q_stride_d2);
        kargs.push_back(q_stride_d3);

        kargs.push_back(k_stride_d0);
        kargs.push_back(k_stride_d1);
        kargs.push_back(k_stride_d2);
        kargs.push_back(k_stride_d3);

        kargs.push_back(v_stride_d0);
        kargs.push_back(v_stride_d1);
        kargs.push_back(v_stride_d2);
        kargs.push_back(v_stride_d3);

        kargs.push_back(output_stride_d0);
        kargs.push_back(output_stride_d1);
        kargs.push_back(output_stride_d2);
        kargs.push_back(output_stride_d3);

        const int grid  = B * H * S * 2;
        // const unsigned int grid_size = static_cast<unsigned>(batch_size) * head_num * q_sequence_length * 2u;

        const int block = 128;

        auto [start, stop] = ctx.get_perf_events();

        k.launch(ctx.get_stream().get(), grid, block, kargs, start, stop);
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
