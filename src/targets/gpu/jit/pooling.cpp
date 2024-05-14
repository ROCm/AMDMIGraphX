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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/op/common.hpp>

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const pooling_kernel = R"__migraphx__(
#include <migraphx/kernels/pooling.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void pooling_kernel(void* in_data, void* output) 
{
    transform_args(make_tensors(), rotate_last())(in_data, output)([](auto&&... xs) { 
        pooling<${count_include_pad}>(${op}, make_window(index_ints<${window}>{}, index_ints<${stride}>{}, index_ints<${padding}>{}), xs...); 
    });
}

}

} // namespace migraphx

)__migraphx__";

struct pooling_compiler : compiler<pooling_compiler>
{
    std::vector<std::string> names() const { return {"pooling"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "pooling_kernel";
        options.virtual_inputs = inputs;

        auto ndim      = out_s.ndim();
        auto pool_ndim = ndim - 2;

        auto read_value = [&](const std::string& name, std::size_t def) {
            if(v.contains(name))
            {
                std::vector<std::size_t> result(2, def);
                auto x = v.at(name).to_vector<std::size_t>();
                if(x.size() >= pool_ndim)
                    result.insert(result.end(), x.begin(), x.begin() + pool_ndim);
                return result;
            }
            else
            {
                std::vector<std::size_t> result(ndim, def);
                return result;
            }
        };

        auto padding = read_value("padding", 0);
        auto stride  = read_value("stride", 1);
        auto window  = read_value("lengths", 1);

        const auto& mode_v = v.at("mode");
        std::string mode =
            mode_v.is_string() ? mode_v.get_string() : to_string(mode_v.to<op::pooling_mode>());

        std::string op = mode + "_pool";
        if(mode == "lpnorm")
            op += "<" + v.at("lp_order").to<std::string>() + ">";

        std::string count_include_pad = v.get("count_include_pad", false) ? "true" : "false";

        auto src = interpolate_string(pooling_kernel,
                                      {{"count_include_pad", count_include_pad},
                                       {"op", op + "{}"},
                                       {"window", to_string_range(window)},
                                       {"stride", to_string_range(stride)},
                                       {"padding", to_string_range(padding)}});

        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
