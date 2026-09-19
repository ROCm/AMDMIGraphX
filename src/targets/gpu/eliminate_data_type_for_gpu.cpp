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
#include <migraphx/gpu/eliminate_data_type_for_gpu.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static void insert_miopen_pooling([[maybe_unused]] std::set<std::string>& u)
{
#if MIGRAPHX_USE_MIOPEN
    u.insert("pooling");
#endif
}

static void insert_gemm_conv(std::set<std::string>& u)
{
    u.insert("convolution");
    u.insert("quant_convolution");
    u.insert("dot");
    u.insert("quant_dot");
}

static eliminate_data_type for_device_functions()
{
    std::set<shape::type_t> unsupported_types(shape::types().begin(), shape::types().end());
    unsupported_types.erase(shape::float_type);
    unsupported_types.erase(shape::half_type);
    unsupported_types.erase(shape::bool_type);
    unsupported_types.erase(shape::int8_type);
    unsupported_types.erase(shape::uint8_type);
    unsupported_types.erase(shape::int32_type);
    unsupported_types.erase(shape::bf16_type);
    unsupported_types.erase(shape::tuple_type);

    std::set<std::string> device_functions = {
        "nonzero",
        "prefix_scan_sum",
        "rnn_var_sl_shift_output",
        "multinomial",
        "argmax",
        "argmin",
    };

    return eliminate_data_type{unsupported_types, shape::float_type, device_functions};
}

static eliminate_data_type for_fp8fnuz(const context* ctx)
{
    std::set<std::string> unsupported_ops = {};

    if(not hipblaslt_supported(*ctx))
    {
        unsupported_ops.insert("dot");
        unsupported_ops.insert("quant_dot");
    }

    insert_miopen_pooling(unsupported_ops);

    if(not gfx_has_fp8fnuz_intrinsics(*ctx))
    {
        insert_gemm_conv(unsupported_ops);
    }
    return eliminate_data_type{
        {shape::fp8e4m3fnuz_type, shape::fp8e5m2fnuz_type}, shape::float_type, unsupported_ops};
}

static eliminate_data_type for_fp8ocp(const context* ctx)
{
    std::set<std::string> unsupported_ops = {};

    if(not hipblaslt_supported(*ctx))
    {
        unsupported_ops.insert("dot");
        unsupported_ops.insert("quant_dot");
    }

    insert_miopen_pooling(unsupported_ops);

    if(not gfx_has_fp8ocp_intrinsics(*ctx))
    {
        insert_gemm_conv(unsupported_ops);
    }
    return eliminate_data_type{
        {shape::fp8e4m3fn_type, shape::fp8e5m2_type}, shape::float_type, unsupported_ops};
}

static eliminate_data_type for_gemm_conv()
{
    std::set<std::string> unsupported_ops = {};
    insert_gemm_conv(unsupported_ops);

    return eliminate_data_type{{
                                   shape::bool_type,
                                   shape::uint16_type,
                                   shape::int16_type,
                                   shape::int64_type,
                                   shape::uint64_type,
                                   shape::double_type,
                               },
                               shape::float_type,
                               unsupported_ops};
}

static void materialize_returned_slice(module& m)
{
    std::vector<std::pair<instruction_ref, instruction_ref>> returned_slices;
    for(auto ins : iterator_for(m))
    {
        if(not contains({"slice", "dyn_slice"}, ins->name()) or ins->inputs().size() < 2)
            continue;

        auto data = ins->inputs().front();
        if(data->name() != "get_tuple_elem" or
           data->inputs().front()->get_shape().type() != shape::tuple_type)
            continue;

        const auto& inputs = ins->inputs();
        if(std::none_of(inputs.begin(), inputs.end(), [](auto input) {
               return contains({shape::int64_type, shape::uint64_type}, input->get_shape().type());
           }))
            continue;

        for(auto output : ins->outputs())
        {
            if(output->name() == "@return")
                returned_slices.emplace_back(ins, output);
        }
    }

    for(auto [slice, output] : returned_slices)
    {
        auto materialized = m.insert_instruction(
            output,
            make_op("convert", {{"target_type", to_value(slice->get_shape().type())}}),
            slice);
        instruction::replace_argument(output, slice, materialized);
    }
}

void eliminate_data_type_for_gpu::apply(module_pass_manager& mpm) const
{
    std::set<shape::type_t> unsupported_floats;
    // No BF-16 Support on Navi21
    if(not gfx_has_bf16_intrinsics(*ctx))
    {
        unsupported_floats.insert(shape::bf16_type);
    }
    if(disable_64bit)
    {
        unsupported_floats.insert(shape::double_type);
    }
    if(not unsupported_floats.empty())
        mpm.run_pass(eliminate_data_type{unsupported_floats, shape::float_type});

    if(disable_64bit)
    {
        // TODO: Check for large tensors
        mpm.run_pass(eliminate_data_type{{shape::int64_type}, shape::int32_type});
        mpm.run_pass(eliminate_data_type{{shape::uint64_type}, shape::uint32_type});
        materialize_returned_slice(mpm.get_module());
    }

    // workaround for rocBLAS unsupported error when using uint8 in quant_dot, quant_convolution &
    // pooling
    mpm.run_pass(eliminate_data_type{
        {shape::uint8_type}, shape::float_type, {"quant_convolution", "quant_dot", "pooling"}});

    mpm.run_pass(for_device_functions());

    mpm.run_pass(for_fp8fnuz(ctx));
    mpm.run_pass(for_fp8ocp(ctx));

    mpm.run_pass(for_gemm_conv());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
