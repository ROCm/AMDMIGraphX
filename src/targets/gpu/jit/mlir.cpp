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
#include <migraphx/gpu/mlir.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct mlir_compiler : compiler<mlir_compiler>
{
    std::vector<std::string> names() const { return {"gpu::mlir_op"}; }

    operation compile_op(context&, const std::vector<shape>&, const value&) const { return {}; }

    compiler_replace
    compile(const context& ctx, instruction_ref ins, const operation&, const value& solution) const
    {
        auto* smod = ins->module_inputs().front();
        assert(smod->get_parameter_names().size() == ins->inputs().size() - 1);
        return insert(compile_mlir(ctx, *smod, ins->inputs(), solution));
    }

    compiler_replace insert(code_object_op co) const
    {
        return {std::vector<operation>{std::move(co)},
                [](module& m, instruction_ref ins, const std::vector<operation>& op) {
                    auto mlir =
                        insert_mlir(m, ins, any_cast<code_object_op>(op.front()), ins->inputs());
                    m.replace_instruction(ins, mlir);
                }};
    }

    optional<tuning_config> get_tuning_config(const context& ctx,
                                              instruction_ref ins,
                                              const operation&,
                                              bool exhaustive) const
    {
        auto shapes = to_shapes(ins->inputs());
        auto* smod  = ins->module_inputs().front();
        return get_tuning_config_mlir(ctx, *smod, shapes, exhaustive);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
