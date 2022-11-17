#include <migraphx/gpu/fuse_ck.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_CK_GEMM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_CK_GEMM_FUSION);

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
        // if(mods.size() != 1)
        //     MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto a = inputs[0];
        auto b = inputs[1];
        for(const auto& input : inputs)
            check_gemm_shape(input);
        return op.compute_shape({a, b});
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm);

// struct ck_gemm_scale_bias_softmax_gemm
// {
//     operation op = make_op("dot");

//     template <class Self, class F>
//     static auto reflect(Self& self, F f)
//     {
//         return pack(f(self.op, "op"));
//     }

//     std::string name() const { return "gpu::ck_gemm_softmax_gemm"; }

//     void check_gemm_shape(const shape& s) const
//     {
//         if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
//             MIGRAPHX_THROW("Invalid shape for ck_gemm_scale_bias_softmax_gemm");
//     }

//     shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
//     {
//         check_shapes{inputs, *this}.same_ndims();
//         // if(mods.size() != 1)
//         //     MIGRAPHX_THROW("should have one submodule.");
//         if(inputs.size() < 2)
//             MIGRAPHX_THROW("should have at least two inputs.");
//         auto a  = inputs[0];
//         auto b  = inputs[1];
//         auto b1 = inputs[2];
//         for(const auto& input : inputs)
//         {
//             // std::cout << input << std::endl;
//             check_gemm_shape(input);
//         }
//         return op.compute_shape({op.compute_shape({a, b}), b1});
//     }
// };
// MIGRAPHX_REGISTER_OP(ck_gemm_scale_bias_softmax_gemm);

namespace {

MIGRAPHX_PRED_MATCHER(is_ck_gemm, instruction_ref ins)
{
    if(ins->name() != "dot")
        return false;
    auto a = ins->inputs().front()->get_shape();
    auto b = ins->inputs().back()->get_shape();
    if(a.lens().back() > 2048)
        return false;
    return true;
}

struct find_ck_gemm_pointwise
{
    // Find a gemm followed by a pointwise operation.
    auto matcher() const
    {
        auto gemm =
            match::skip(match::name("contiguous"))(match::name("dot")(is_ck_gemm().bind("gemm")));
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
        assert(gemm_it != inputs.end());
        if(ins->get_shape().type() != shape::half_type)
            return;
        if(gemm_idx != 0)
        {
            auto first_param    = pm->get_parameter(names[0]);
            auto gemm_param     = pm->get_parameter(names[gemm_idx]);
            auto new_gemm_param = pm->add_parameter(names[0] + ".0", gemm_param->get_shape());
            auto new_first_param =
                pm->add_parameter(names[gemm_idx] + ".0", first_param->get_shape());
            pm->replace_instruction(gemm_param, new_gemm_param);
            pm->replace_instruction(first_param, new_first_param);
            pm->remove_instruction(first_param);
            pm->remove_instruction(gemm_param);
        }
        inputs.erase(gemm_it);
        inputs.insert(inputs.begin(), gemm_ins->inputs().begin(), gemm_ins->inputs().end());

        mpm.get_module().replace_instruction(ins, ck_gemm{}, inputs, {pm});
    }
};

struct find_ck_gemm
{
    auto matcher() const { return match::name("dot")(is_ck_gemm().bind("gemm")); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        mpm.get_module().replace_instruction(ins, ck_gemm{ins->get_operator()}, ins->inputs());
    }
};

struct find_ck_gemm_scale_bias_softmax_gemm
{
    // auto matcher() const
    // {
    //     auto gemm1 =
    //         match::skip(match::name("contiguous"))(match::name("dot")(is_ck_gemm().bind("gemm1")));
    //     auto pw =
    //         match::name("pointwise")(match::any_of[match::inputs()](gemm1)).bind("scale_bias");
    //     auto softmax =
    //     match::name("softmax")(match::any_of[match::inputs()](pw)).bind("softmax"); return
    //     match::name("dot")(is_ck_gemm().bind("gemm2"))(
    //         match::any_of[match::inputs()](softmax));
    // }

    // void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    // {
    //     std::cout << "Matched" << std::endl;
    //     auto ins       = r.result;
    //     auto gemm2_ins = r.instructions["gemm2"];
    //     auto sm_ins    = r.instructions["softmax"];
    //     auto pw_ins    = r.instructions["scale_bias"];
    //     auto gemm1_ins = r.instructions["gemm1"];

    //     gemm2_ins->debug_print();
    //     sm_ins->debug_print();
    //     pw_ins->debug_print();
    //     gemm1_ins->debug_print();

    //     auto inputs = gemm1_ins->inputs();            // A, B
    //     inputs.push_back(gemm2_ins->inputs().back()); // B1
    //     // inputs.push_back(pw_ins->inputs().back()); // C

    //     mpm.get_module().replace_instruction(
    //         ins, ck_gemm_scale_bias_softmax_gemm{gemm2_ins->get_operator()}, inputs);
    // }

    // auto matcher() const
    // {
    //     auto gemm1 =
    //     match::skip(match::name("contiguous"))(match::name("dot")(is_ck_gemm().bind("gemm1")));
    //     auto softmax =
    //     match::name("softmax")(match::any_of[match::inputs()](gemm1)).bind("softmax"); return
    //     match::name("dot")(is_ck_gemm().bind("gemm2"))(match::any_of[match::inputs()](softmax));
    // }

    // void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    // {
    //     std::cout << "Matched" << std::endl;
    //     auto ins = r.result;
    //     auto gemm2_ins = r.instructions["gemm2"];
    //     auto sm_ins = r.instructions["softmax"];
    //     auto gemm1_ins = r.instructions["gemm1"];

    //     gemm2_ins->debug_print();
    //     sm_ins->debug_print();
    //     gemm1_ins->debug_print();

    //     auto inputs = gemm1_ins->inputs(); // A, B
    //     inputs.push_back(gemm2_ins->inputs().back()); // B1

    //     mpm.get_module().replace_instruction(ins,
    //     ck_gemm_scale_bias_softmax_gemm{gemm2_ins->get_operator()}, inputs);
    // }
};

} // namespace

void fuse_ck::apply(module_pass_manager& mpm) const
{
    // mpm.get_module().debug_print();
    // match::find_matches(mpm, find_ck_gemm_scale_bias_softmax_gemm{});
    if(not enabled(MIGRAPHX_DISABLE_CK_GEMM_FUSION{}))
        match::find_matches(mpm, find_ck_gemm_pointwise{});
    if(not enabled(MIGRAPHX_DISABLE_CK_GEMM{}))
        match::find_matches(mpm, find_ck_gemm{});
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
