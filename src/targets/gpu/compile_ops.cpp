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
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/gpu/compiler.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_COMPILE_PARALLEL);

struct precompile_op
{
    operation op                = op::identity{};
    std::size_t additional_args = 1;
    bool ignore_modules         = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"),
                    f(self.additional_args, "additional_args"),
                    f(self.ignore_modules, "ignore_modules"));
    }

    std::string name() const { return "gpu::precompile_op"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        // Pop off additional args
        inputs.resize(inputs.size() - additional_args);
        if(ignore_modules)
            return op.compute_shape(inputs);
        return op.compute_shape(inputs, mods);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

MIGRAPHX_REGISTER_OP(precompile_op);

struct compiled_result
{
    compiler_replace replace;
    instruction_ref ins;
};

template <class F>
void par_compile(std::size_t n, F f)
{
    if(n == 0)
        return;
    par_for(n, n / value_of(MIGRAPHX_GPU_COMPILE_PARALLEL{}, n), f);
}

void compile_ops::apply(module& m) const
{
    std::vector<std::function<compiled_result()>> compiles;

    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::precompile_op")
            continue;
        operation preop = any_cast<precompile_op>(ins->get_operator()).op;
        compiles.emplace_back([=]() -> compiled_result {
            return {compile(*ctx, ins, preop), ins};
        });
    }
    std::vector<compiled_result> results(compiles.size());
    par_compile(compiles.size(), [&](auto i) { results[i] = compiles[i](); });
    for(const auto& cr : results)
    {
        cr.replace(m, cr.ins);
    }
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
