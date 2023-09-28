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
#include <migraphx/gpu/fuse_ck.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

struct ck_gemm
{
    operation op = make_op("dot");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::ck_gemm"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.same_ndims();
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto a = inputs[0];
        auto b = inputs[1];
        for(const auto& input : inputs)
            check_gemm_shape(input);
        auto r = op.compute_shape({a, b});
        if(mods.empty())
            return r;
        return r.with_type(mods.front()->get_output_shapes().front().type());
    }

    static bool is_ck_supported_type(shape::type_t t)
    {
        return contains({shape::half_type, shape::int8_type, shape::int32_type}, t);
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm);

struct ck_gemm_softmax_gemm
{
    operation op = make_op("dot");
    double scale = 1.0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"), f(self.scale, "scale"));
    }

    std::string name() const { return "gpu::ck_gemm_softmax_gemm"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm_softmax_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>&) const
    {
        check_shapes{inputs, *this}.same_ndims();
        if(inputs.size() < 3)
            MIGRAPHX_THROW("Expected 3 inputs but got " + to_string(inputs.size()));
        auto a  = inputs[0];
        auto b  = inputs[1];
        auto b1 = inputs[2];
        for(const auto& input : inputs)
        {
            check_gemm_shape(input);
        }
        return op.compute_shape({op.compute_shape({a, b}), b1});
    }

    static bool is_ck_supported_type(shape::type_t t) { return contains({shape::half_type}, t); }
};
MIGRAPHX_REGISTER_OP(ck_gemm_softmax_gemm);

namespace {

MIGRAPHX_PRED_MATCHER(is_ck_gemm, instruction_ref ins)
{
    if(ins->name() != "dot" and ins->name() != "quant_dot")
        return false;
    if(not ck_gemm::is_ck_supported_type(ins->get_shape().type()))
        return false;
    auto a = ins->inputs().front()->get_shape();
    auto b = ins->inputs().back()->get_shape();
    auto m = a.lens()[a.lens().size() - 2];
    auto n = b.lens().back();
    auto k = a.lens().back();
    // Integer gemms must be divisible by 4 in ck
    if(contains({shape::int8_type, shape::int32_type}, ins->get_shape().type()))
    {
        if(m % 4 != 0)
            return false;
        if(n % 4 != 0)
            return false;
        if(k % 4 != 0)
            return false;
    }
    // Skipping GEMMs with a K dimension greater than 2048 is a course-grained strategy
    // to avoid poor-performing GEMM kernels from CK
    // To-do: Investigate a more precise strategy
    return k <= 2048;
}

struct find_ck_gemm_pointwise
{
    // Find a gemm followed by a pointwise operation.
    auto matcher() const
    {
        auto gemm = match::skip(match::name("contiguous"))(
            match::name("dot", "quant_dot")(is_ck_gemm().bind("gemm")));
        return match::name("pointwise")(match::any_of[match::inputs()](gemm.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["gemm"];
        auto x_ins    = r.instructions["x"]; // input after contiguous
        auto* pm      = ins->module_inputs().front();
        auto names    = pm->get_parameter_names();
        std::sort(names.begin(), names.end());
        auto inputs   = ins->inputs();
        auto gemm_it  = std::find(inputs.begin(), inputs.end(), x_ins);
        auto gemm_idx = gemm_it - inputs.begin();
        if(gemm_ins->get_shape().type() != shape::int32_type and
           ins->get_shape().type() != gemm_ins->get_shape().type())
            return;
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [](auto input) {
               return not ck_gemm::is_ck_supported_type(input->get_shape().type());
           }))
            return;
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [](auto input) {
               return not input->inputs().empty() and input->inputs().front()->name() == "capture";
           }))
            return;
        assert(gemm_it != inputs.end());
        if(gemm_idx != 0)
        {
            auto first_param    = pm->get_parameter(names[0]);
            auto gemm_param     = pm->get_parameter(names[gemm_idx]);
            auto new_gemm_param = pm->add_parameter(names[0] + "_0", gemm_param->get_shape());
            auto new_first_param =
                pm->add_parameter(names[gemm_idx] + "_0", first_param->get_shape());
            pm->replace_instruction(gemm_param, new_gemm_param);
            pm->replace_instruction(first_param, new_first_param);
            pm->remove_instruction(first_param);
            pm->remove_instruction(gemm_param);
        }
        inputs.erase(gemm_it);
        inputs.insert(inputs.begin(), gemm_ins->inputs().begin(), gemm_ins->inputs().end());

        mpm.get_module().replace_instruction(ins, ck_gemm{gemm_ins->get_operator()}, inputs, {pm});
    }
};

struct find_ck_gemm
{
    auto matcher() const { return match::name("dot", "quant_dot")(is_ck_gemm().bind("gemm")); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        mpm.get_module().replace_instruction(ins, ck_gemm{ins->get_operator()}, ins->inputs());
    }
};

static auto is_mul_module(module& m)
{
    auto is_mul =
        match::arg(0)(match::name("mul")(match::all_of[match::inputs()](match::name("@param"))));
    return match_instruction(m, std::prev(m.end()), is_mul).result != m.end();
}

MIGRAPHX_PRED_MATCHER(is_pointwise_scale, instruction_ref ins)
{
    if(ins->name() != "pointwise")
        return false;
    if(ins->module_inputs().size() != 1)
        return false;
    return is_mul_module(*ins->module_inputs().front());
}

struct find_ck_gemm_softmax_gemm
{
    auto matcher() const
    {
        auto gemm1 =
            match::skip(match::name("contiguous"))(match::name("dot")(is_ck_gemm().bind("gemm1")));
        auto mul     = match::name("pointwise")(match::either_arg(0, 1)(
            match::is_constant().bind("scale"), gemm1))(is_pointwise_scale());
        auto softmax = match::name("softmax")(match::arg(0)(mul)).bind("softmax");

        return match::name("dot")(is_ck_gemm().bind("gemm2"))(match::arg(0)(softmax));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm2_ins = r.instructions["gemm2"];
        auto gemm1_ins = r.instructions["gemm1"];
        auto scale_lit = r.instructions["scale"];

        if(not ck_gemm_softmax_gemm::is_ck_supported_type(gemm1_ins->get_shape().type()))
            return;

        double scale = 1.0;
        scale_lit->eval().visit([&](const auto s) {
            // CK only supports single-valued scale
            if(std::all_of(
                   s.begin() + 1, s.end(), [&](auto v) { return float_equal(v, s.front()); }))
                scale = s.front();
            else
                return;
        });

        auto inputs = gemm1_ins->inputs();            // A, B
        inputs.push_back(gemm2_ins->inputs().back()); // B1

        mpm.get_module().replace_instruction(
            ins, ck_gemm_softmax_gemm{gemm2_ins->get_operator(), scale}, inputs);
    }
};

} // namespace

void fuse_ck::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_ck_gemm_softmax_gemm{});
    match::find_matches(mpm, find_ck_gemm_pointwise{});
    match::find_matches(mpm, find_ck_gemm{});
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
