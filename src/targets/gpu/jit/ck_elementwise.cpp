/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/compile_gen.hpp>

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// static const char* const ck_elementwise_kernel = R"__migraphx__(
// //#include <migraphx/kernels/ck_elementwise.hpp>
// #include <migraphx/kernels/ops.hpp>
// #include <migraphx/kernels/integral_constant.hpp>
// #include <migraphx/kernels/generic_constant.hpp>
// #include <args.hpp>

// #include <migraphx/kernels/index.hpp>
// #include <migraphx/kernels/algorithm.hpp>
// #include <migraphx/kernels/integral_constant.hpp>
// #include <migraphx/kernels/tensor_view.hpp>

// #include "ck/device_utility/device_prop.hpp"
// #include "ck/device_utility/kernel_launch.hpp"
// #include "ck/tensor_operation/gpu/device/device_base.hpp"
// #include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
// #include "ck/tensor_operation/gpu/grid/gridwise_binary_elementwise_1d.hpp"

// namespace migraphx {

// using ADataType          = float;
// using BDataType          = float;
// using CDataType          = float;
// using ElementwiseFunctor = float;

// static constexpr auto I0 = ck::Number<0>{};

// template <class L, class S, class N>
// constexpr auto MakeDescriptor_M(const L& lengths, const S& strides, const N& ndim)
// {
//     auto gridSize       = 72;
//     auto blockSize      = 1024;
//     //constexpr auto ndim = 1;
//     // auto idx          = make_index();
//     auto tupleOfShape = generate_tuple([&](auto I) { return static_cast<ck::index_t>(lengths[I]);
//     },
//                                        ck::Number<ndim>{});
//     auto tupleOfStride = generate_tuple(
//         [&](auto I) { return static_cast<ck::index_t>(strides[I]); }, ck::Number<1>{});
//     const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);
//     auto desc_m     = desc;
//     // merge nd to 1d desc - [s0 * s1 * ...]
//     if constexpr(ndim > 1)
//     {
//         desc_m = transform_tensor_descriptor(
//             desc,
//             make_tuple(make_merge_transform(tupleOfShape)),
//             make_tuple(generate_sequence_v2([&](auto I) { return I; }, ck::Number<ndim>{})),
//             make_tuple(ck::Sequence<0>{}));
//     }

//     const auto M                = desc_m.GetLength(I0);
//     const ck::index_t loop_step = /* idx.nglobal(); // */ gridSize * blockSize /*  * MPerThread
//     */; const auto pad              = ck::math::integer_least_multiple(M, loop_step) - M; const
//     auto desc_m_pad =
//         transform_tensor_descriptor(desc_m,
//                                     make_tuple(ck::make_right_pad_transform(M, pad)),
//                                     make_tuple(ck::Sequence<0>{}),
//                                     make_tuple(ck::Sequence<0>{}));
//     return desc_m_pad;
// }

// struct Add
// {
//     template <typename Y, typename X0, typename X1>
//     __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const
//     {
//         y = x0 + x1;
//     };
// };

// extern "C" {

// __global__ void ck_elementwise_kernel(void* a_p, void* b_p, void* c_p)
// {
//     make_tensors()(a_p, b_p, c_p)([](auto a_t, auto b_t, auto c_t) {
//         constexpr auto lengths = get_shape_c<decltype(a_t)>{}.lens;
//         constexpr auto strides = get_shape_c<decltype(a_t)>{}.strides;
//         constexpr auto ndim = _c<decltype(lengths.size()){}>[1];
//         constexpr auto a_desc  = MakeDescriptor_M(lengths, strides, ndim);

//         using AGridDesc_M        = decltype(a_desc);
//         using GridwiseBinEltwise = ck::GridwiseBinaryElementwise_1D<ADataType,
//                                                                     BDataType,
//                                                                     CDataType,
//                                                                     CDataType,
//                                                                     AGridDesc_M,
//                                                                     AGridDesc_M,
//                                                                     AGridDesc_M,
//                                                                     Add,
//                                                                     1,
//                                                                     1,
//                                                                     1,
//                                                                     1>;
//         auto op                  = Add{};
//         GridwiseBinEltwise::Run(a_t.data(), b_t.data(), c_t.data(), a_desc, a_desc, a_desc, op);
//     });
// }

// }

// } // namespace migraphx

// )__migraphx__";

// NOLINTNEXTLINE
static const char* const ck_elementwise_kernel = R"__migraphx__(
#include <migraphx/kernels/ck_elementwise.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

__global__ void ck_elementwise_kernel(void* a_p, void* b_p, void* c_p)
{
    make_tensors()(a_p, b_p, c_p)([](auto&&... xs) {
        ck_elementwise(xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct ck_elementwise_compiler : compiler<ck_elementwise_compiler>
{
    std::vector<std::string> names() const { return {"ck_elementwise"}; }

    static std::size_t oversubscribe_if(bool b)
    {
        if(b)
            return 256;
        else
            return 1;
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // hip_compile_options options;
        // auto out_s = inputs.back();
        // options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        // options.inputs         = inputs;
        // options.output         = out_s;
        // options.kernel_name    = "ck_elementwise_kernel";
        // options.virtual_inputs = inputs;

        // return compile_hip_code_object(ck_elementwise_kernel, options);
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(inputs);
        options.params         = "-Wno-float-equal";
        auto axis              = find_fast_axis(options.virtual_inputs);
        auto vec               = vectorize::elements(axis, options.virtual_inputs);
        auto preloads          = preload::broadcasts(axis, options.virtual_inputs);
        options.kernel_name    = "ck_elementwise_kernel";
        options.set_launch_params(
            v,
            compute_global_for(ctx,
                               options.output.elements() / vec.size,
                               oversubscribe_if(not preloads.is_preloading())));
        return compile_hip_code_object(ck_elementwise_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
