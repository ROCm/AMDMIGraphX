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
#include <fstream>
#include <migraphx/filesystem.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/ck.hpp>
#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const debug_op_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/debug_op.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/ops.hpp>


namespace migraphx {



extern "C" {


MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_and_pack_last<${noutputs}>())(${args})([](auto... xs) {
        debug_op(xs...);
    });
}


}

} // namespace migraphx

)__migraphx__";

struct debug_op_compiler : compiler<debug_op_compiler>
{
    std::vector<std::string> names() const { return {"debug_op", "gpu::debug_op"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        for (int i = 0; i < inputs.size(); ++i)
        {
            std::cout << inputs[i] << std::endl;
        }
        std::cout << "___________" << std::endl;
        auto virtual_inputs = inputs;
        virtual_inputs = flatten(virtual_inputs);
        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, virtual_inputs.front().elements()));
        int blocks_per_batch = 1; ////
        options.inputs         = virtual_inputs;
        options.output         = inputs.back();//output_shape;
        options.kernel_name    = v.get("kernel", "debug_op_kernel");

        if(v.get("check", false) or enabled(MIGRAPHX_CK_DEBUG{}))
            options.emplace_param("-DMIGRAPHX_CK_CHECK=1");
        // std::cout << "gqa compile 3" << std::endl;
        for (int i = 0; i < virtual_inputs.size(); ++i)
        {
            std::cout << virtual_inputs[i] << std::endl;
        }
        auto src = interpolate_string(debug_op_kernel,
                                      {
                                        {"params", enum_params(virtual_inputs.size(), "void * private_p")},
                                       {"args", enum_params(virtual_inputs.size(), "private_p")},
                                       {"blocks_per_batch", to_string(blocks_per_batch)},
                                       {"kernel", options.kernel_name},
                                       {"noutputs", std::to_string(3)}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto shapes = to_shapes(ins->inputs());
        auto v = op.to_value();
        return compile_op(ctx, shapes, v);
    }

};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
