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

#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/gpu/fuse_ck.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/gpu/device_name.hpp>

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
            MIGRAPHX_THROW(name() + ": should have at least two inputs.");
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

struct ck_gemm_gemm
{
    operation op     = make_op("dot");
    size_t d0s_count = 0u;
    size_t d1s_count = 0u;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.op, "op"), f(self.d0s_count, "d0s_count"), f(self.d1s_count, "d1s_count"));
    }

    std::string name() const { return "gpu::ck_gemm_gemm"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1) and
           not s.scalar())
            MIGRAPHX_THROW("Invalid shape for " + name());
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>&) const
    {
        check_shapes{inputs, *this}.same_ndims();
        if(inputs.size() < 3)
            MIGRAPHX_THROW(name() + ": Expected 3 inputs but got " + to_string(inputs.size()));

        auto a  = inputs[0];
        auto b  = inputs[1];
        auto b1 = inputs[2];

        for(const auto& input : inputs)
        {
            check_gemm_shape(input);
        }

        auto gemm0_shape = op.compute_shape({a, b});
        return op.compute_shape({gemm0_shape, b1});
    }

    static bool is_ck_supported_type(shape::type_t t) { return contains({shape::half_type}, t); }
};
MIGRAPHX_REGISTER_OP(ck_gemm_gemm);

struct ck_gemm_softmax_gemm : gemm_softmax_gemm
{
    std::string name() const { return "gpu::ck_gemm_softmax_gemm"; }
};
MIGRAPHX_REGISTER_OP(ck_gemm_softmax_gemm);

namespace {

MIGRAPHX_PRED_MATCHER(is_ck_gemm, instruction_ref ins)
{
    if(ins->name() != "dot" and ins->name() != "quant_dot")
        return false;
    if(not ck_gemm::is_ck_supported_type(ins->get_shape().type()))
        return false;
    auto a          = ins->inputs().front()->get_shape();
    auto b          = ins->inputs().back()->get_shape();
    auto m          = a.lens()[a.lens().size() - 2];
    auto n          = b.lens().back();
    auto k          = a.lens().back();
    auto batch_size = std::accumulate(
        a.lens().rbegin() + 2, a.lens().rend(), std::size_t{1}, std::multiplies<std::size_t>());
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
    auto device_name = trim(split_string(get_device_name(), ':').front());
    if(starts_with(device_name, "gfx94"))
    {
        if(ins->get_shape().type() == shape::half_type)
        {
            if(batch_size >= 64)
                return m < 2048 or k <= 64 or n <= 384 or n >= 2048;
            return true;
        }
        return true;
    }
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

struct find_ck_gemm_softmax_gemm
{
    auto matcher() const { return match::name("gpu::pre_gemm_softmax_gemm"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto v   = ins->get_operator().to_value();
        assert(v.contains("scale"));
        auto scale = v.at("scale").to<float>();
        mpm.get_module().replace_instruction(
            ins, ck_gemm_softmax_gemm{migraphx::make_op("dot"), scale}, ins->inputs());
    }
};

struct find_ck_gemm_pointwise_gemm
{
    auto matcher() const
    {
        // TODO don't mix dot and quant_dot
        // TODO match used_once?
        auto gemm0 = match::skip(match::name("contiguous"))(
            match::name("dot", "quant_dot")(is_ck_gemm().bind("gemm0")));
        auto pw0 =
            match::name("pointwise")(match::any_of[match::inputs()](gemm0.bind("x0")).bind("pw0"));
        return match::name("dot", "quant_dot")(is_ck_gemm().bind("gemm1"))(
            match::arg(0)(match::any_of(pw0, gemm0)));
    }

    bool transposed_matrix(const shape& s) { return s.strides().back() != 1; }

    void apply(module_pass_manager& mpm, const match::matcher_result& r)
    {
        auto ins       = r.result;
        auto gemm0_ins = r.instructions["gemm0"];
        auto gemm1_ins = r.instructions["gemm1"];

        auto inputs = gemm0_ins->inputs();            // A, B
        inputs.push_back(gemm1_ins->inputs().back()); // B1

        if (!transposed_matrix(inputs[1]->get_shape()))
            return;

        size_t d0s_count = 0, d1s_count = 0;
        std::vector<module_ref> module_inputs;
        if(r.instructions.find("pw0") != r.instructions.end())
        {
            auto pw0    = r.instructions["pw0"];
            auto x0_ins = r.instructions["x0"];

            if(gemm0_ins->get_shape().type() != shape::int32_type and
               pw0->get_shape().type() != gemm0_ins->get_shape().type())
                return;
            if(std::any_of(pw0->inputs().begin(), pw0->inputs().end(), [](auto input) {
                   return not ck_gemm::is_ck_supported_type(input->get_shape().type());
               }))
                return;
            if(std::any_of(pw0->inputs().begin(), pw0->inputs().end(), [](auto input) {
                   return not input->inputs().empty() and
                          input->inputs().front()->name() == "capture";
               }))
                return;

            auto pw0_inputs = pw0->inputs();
            auto* pw0m      = pw0->module_inputs().front();

            auto gemm_it  = std::find(pw0_inputs.begin(), pw0_inputs.end(), x0_ins);
            auto gemm_idx = gemm_it - pw0_inputs.begin();
            if(gemm_idx != 0)
            {
                rotate_gemm_input(pw0m, gemm_idx);
            }

            pw0_inputs.erase(gemm_it);
            inputs.insert(inputs.end(), pw0_inputs.begin(), pw0_inputs.end()); // D0s

            d0s_count = pw0_inputs.size();
            module_inputs.push_back(pw0m);
        }
        if(r.instructions.find("pw1") != r.instructions.end())
        {
            auto pw1 = r.instructions["pw1"];

            if(gemm1_ins->get_shape().type() != shape::int32_type and
               pw1->get_shape().type() != gemm1_ins->get_shape().type())
                return;
            if(std::any_of(pw1->inputs().begin(), pw1->inputs().end(), [](auto input) {
                   return not ck_gemm::is_ck_supported_type(input->get_shape().type());
               }))
                return;
            if(std::any_of(pw1->inputs().begin(), pw1->inputs().end(), [](auto input) {
                   return not input->inputs().empty() and
                          input->inputs().front()->name() == "capture";
               }))
                return;

            auto pw1_inputs = pw1->inputs();
            auto* pw1m      = pw1->module_inputs().front();

            auto gemm_it  = std::find(pw1_inputs.begin(), pw1_inputs.end(), gemm1_ins);
            auto gemm_idx = gemm_it - pw1_inputs.begin();
            if(gemm_idx != 0)
            {
                rotate_gemm_input(pw1m, gemm_idx);
            }

            pw1_inputs.erase(gemm_it);
            inputs.insert(inputs.end(), pw1_inputs.begin(), pw1_inputs.end()); // D1s

            d1s_count = pw1_inputs.size();
            module_inputs.push_back(pw1m);
        }

        mpm.get_module().replace_instruction(
            ins,
            ck_gemm_gemm{gemm1_ins->get_operator(), d0s_count, d1s_count},
            inputs,
            module_inputs);
    }

    void rotate_gemm_input(module* pwm, size_t gemm_idx)
    {
        auto names = pwm->get_parameter_names();

        auto first_param = pwm->get_parameter(names[0]);
        auto gemm_param  = pwm->get_parameter(names[gemm_idx]);

        auto new_gemm_param  = pwm->add_parameter(names[0] + "_0", gemm_param->get_shape());
        auto new_first_param = pwm->add_parameter(names[gemm_idx] + "_0", first_param->get_shape());

        pwm->replace_instruction(gemm_param, new_gemm_param);
        pwm->replace_instruction(first_param, new_first_param);
        pwm->remove_instruction(first_param);
        pwm->remove_instruction(gemm_param);
    }
};

struct find_ck_gemm_pointwise_gemm_pointwise : find_ck_gemm_pointwise_gemm
{
    auto matcher() const
    {
        // TODO match used_once?
        auto gemm1 = find_ck_gemm_pointwise_gemm::matcher();
        return match::name("pointwise")(match::any_of[match::inputs()](gemm1).bind("pw1"));
    }
};

} // namespace

void fuse_ck::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_ck_gemm_softmax_gemm{});
    match::find_matches(mpm, find_ck_gemm_pointwise_gemm_pointwise{});
    match::find_matches(mpm, find_ck_gemm_pointwise_gemm{});
    match::find_matches(mpm, find_ck_gemm_pointwise{});
    match::find_matches(mpm, find_ck_gemm{});
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
