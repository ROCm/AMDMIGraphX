#include <migraphx/gpu/fuse_ck_gemm_softmax_gemm.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

struct gemm_softmax_gemm_gemm
{
    operation op = make_op("dot");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::ck_gemm_softmax_gemm"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
            MIGRAPHX_THROW("Invalid shape for gemm_softmax_gemm_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.same_ndims();
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto a  = inputs[0];
        auto b  = inputs[1];
        auto b1 = inputs[2];
        for(const auto& input : inputs)
        {
            check_gemm_shape(input);
        }
        return op.compute_shape({op.compute_shape({a, b}), b1});
    }
};
MIGRAPHX_REGISTER_OP(gemm_softmax_gemm_gemm);

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

struct find_gemm_softmax_gemm_gemm
{
    auto matcher() const
    {
        auto gemm1 =
            match::skip(match::name("contiguous"))(match::name("dot")(is_ck_gemm().bind("gemm1")));
        auto mul =
            match::name("mul")(match::any_of[match::inputs()](gemm1)).bind("scale");
        auto add =
            match::name("add")(match::any_of[match::inputs()](mul));
        auto softmax = match::name("softmax")(match::any_of[match::inputs()](add)).bind("softmax");
        return match::name("dot")(is_ck_gemm().bind("gemm2"))(
            match::any_of[match::inputs()](softmax));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm2_ins = r.instructions["gemm2"];
        auto gemm1_ins = r.instructions["gemm1"];

        auto inputs = gemm1_ins->inputs();            // A, B
        inputs.push_back(gemm2_ins->inputs().back()); // B1

        mpm.get_module().replace_instruction(
            ins, gemm_softmax_gemm_gemm{gemm2_ins->get_operator()}, inputs);
    }
};

} // namespace

void fuse_ck_gemm_softmax_gemm::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_gemm_softmax_gemm_gemm{});
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
