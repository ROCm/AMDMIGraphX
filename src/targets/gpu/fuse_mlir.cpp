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
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR);

#ifdef MIGRAPHX_MLIR

struct mlir_op
{
    std::string name() const { return "gpu::mlir_op"; }
    operation op = make_op("convolution");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.packed_or_broadcasted();
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto n = inputs.size();
        return op.compute_shape({inputs[n - 2], inputs[n - 1]});
    }
};
MIGRAPHX_REGISTER_OP(mlir_op);

namespace {

MIGRAPHX_PRED_MATCHER(is_mlir_conv, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    value v    = ins->get_operator().to_value();
    auto group = v.at("group").to<int>();
    if(group != 1)
        return false;
    // Avoid MLIR assertion: Index < Length && "Invalid index!"
    if(ins->get_shape().lens().size() != 4)
        return false;
    return true;
}

struct find_mlir_op
{
    auto matcher() const
    {
        auto dot_or_conv = match::skip(match::name("contiguous"))(
            match::any_of(match::name("dot"), is_mlir_conv()).bind("gemm_based_op"));
        return match::name("pointwise")(match::any_of[match::inputs()](dot_or_conv.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins           = r.result;
        auto gemm_based_op = r.instructions["gemm_based_op"];
        auto x_ins         = r.instructions["x"]; // input after contiguous
        auto* pm           = ins->module_inputs().front();
        auto names         = pm->get_parameter_names();
        // Whitelist pointwise operators
        if(std::any_of(pm->begin(), pm->end(), [](const auto& i) {
               return not contains(
                   {"@literal", "@param", "@return", "convolution", "dot", "add", "relu"},
                   i.name());
           }))
            return;
        // Only fuse with fp32/fp16
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
               return not contains({shape::type_t::float_type, shape::type_t::half_type},
                                   i->get_shape().type());
           }))
            return;
        std::sort(names.begin(), names.end());
        module_ref mm = mpm.create_module("mlir_" + pm->name());
        mm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> param_map;
        auto x    = mm->add_parameter("x" + std::to_string(names.size()),
                                   gemm_based_op->inputs().at(0)->get_shape());
        auto w    = mm->add_parameter("x" + std::to_string(names.size() + 1),
                                   gemm_based_op->inputs().at(1)->get_shape());
        auto conv = mm->add_instruction(gemm_based_op->get_operator(), {x, w});
        std::transform(names.begin(),
                       names.end(),
                       ins->inputs().begin(),
                       std::inserter(param_map, param_map.end()),
                       [&](auto name, auto input) {
                           if(input == x_ins)
                               return std::make_pair(pm->get_parameter(name), conv);
                           return std::make_pair(pm->get_parameter(name),
                                                 mm->add_parameter(name, input->get_shape()));
                       });
        mm->add_return(mm->insert_instructions(mm->end(), pm, param_map));

        std::vector<instruction_ref> inputs;
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != gemm_based_op; });
        inputs.insert(inputs.end(), gemm_based_op->inputs().begin(), gemm_based_op->inputs().end());
        mpm.get_module().replace_instruction(
            ins, mlir_op{gemm_based_op->get_operator()}, inputs, {mm});
    }
};

} // namespace

#endif

void fuse_mlir::apply(module_pass_manager& mpm) const
{
#ifdef MIGRAPHX_MLIR
    const bool mlir_enabled = enabled(MIGRAPHX_ENABLE_MLIR{});
    if(mlir_enabled)
    {
        match::find_matches(mpm, find_mlir_op{});
    }
    else
    {
        std::cerr << "WARNING: MIGraphX built with MLIR but it is not enabled. Please set the env "
                     "var MIGRAPHX_ENABLE_MLIR to use MLIR kernel generator."
                  << std::endl;
    }
#else
    (void)mpm;
#endif
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
