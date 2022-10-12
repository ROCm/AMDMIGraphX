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
        if(contains(s.lens(), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.not_broadcasted();
        // if(mods.size() != 1)
        //     MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto n = inputs.size();
        auto a = inputs[n - 2];
        auto b = inputs[n - 1];
        check_gemm_shape(a);
        check_gemm_shape(b);
        return op.compute_shape({a, b});
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm);

struct ck_gemm_add_add_gelu
{
    operation op = make_op("dot");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::ck_gemm_add_add_gelu"; }

    void check_gemm_shape(const shape& s) const
    {
        if(contains(s.lens(), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.not_broadcasted();
        // if(mods.size() != 1)
        //     MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto n = inputs.size();
        auto a = inputs[n - 2];
        auto b = inputs[n - 1];
        check_gemm_shape(a);
        check_gemm_shape(b);
        return op.compute_shape({a, b});
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm_add_add_gelu);

namespace {

MIGRAPHX_PRED_MATCHER(is_ck_gemm, instruction_ref ins)
{
    if(ins->name() != "dot")
        return false;
    auto a = ins->inputs().front()->get_shape();
    auto b = ins->inputs().back()->get_shape();
    if(a.lens().size() > 2 or b.lens().size() > 2)
        return false;
    return (a.lens()[0] % 8 == 0 and a.lens()[1] % 8 == 0 and b.lens()[0] % 8 == 0 and
            b.lens()[1] % 8 == 0);
}

struct find_ck_gemm
{
    // Find a convolution followed by a pointwise operation.
    auto matcher() const { return match::name("dot")(is_ck_gemm().bind("gemm")); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        mpm.get_module().replace_instruction(ins, ck_gemm{ins->get_operator()}, ins->inputs());
    }
};

struct find_ck_gemm_pointwise
{
    auto matcher() const { return match::name("pointwise")(match::arg(0)(match::name("dot")(is_ck_gemm().bind("gemm")))); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        // Currently set to only handle gemm_add_add_gelu fusion
        // To-do: Adapt to handle other gemm-pointwise fusions
        auto ins = r.result;
        auto gemm = r.instructions["gemm"];
        auto inputs = gemm->inputs();
        for (auto in : ins->inputs())
        {
            if (in != gemm)
                inputs.push_back(in);
        }
            
        mpm.get_module().replace_instruction(ins, ck_gemm_add_add_gelu{gemm->get_operator()}, inputs);
        mpm.get_module().remove_instruction(gemm);
    }
};

} // namespace

void fuse_ck::apply(module_pass_manager& mpm) const { match::find_matches(mpm, find_ck_gemm_pointwise{}); }

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
