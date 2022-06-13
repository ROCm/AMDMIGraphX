#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

struct mlir_conv
{
    operation op = make_op("convolution");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::mlir_conv"; }
    shape compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
    {
        check_shapes{inputs, *this}.standard();
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");
        auto n = inputs.size();
        return op.compute_shape({inputs[n - 2], inputs[n - 1]});
    }
};
MIGRAPHX_REGISTER_OP(mlir_conv);

namespace {
struct find_conv_pointwise
{
    // Find a convolution followed by a pointwise operation.
    auto matcher() const
    {
        auto convolution =
            match::skip(match::name("contiguous"))(match::name("convolution").bind("convolution"));
        return match::name("pointwise")(match::any_of[match::inputs()](convolution.bind("x")));
    }

    void apply(module_pass_manager& mpm, match::matcher_result r) const
    {
        auto ins      = r.result;
        auto conv_ins = r.instructions["convolution"];
        auto x_ins    = r.instructions["x"]; // input after contiguous
        auto pm       = ins->module_inputs().front();
        auto names    = pm->get_parameter_names();
        // Whitelist pointwise operators
        if(std::any_of(pm->begin(), pm->end(), [](const auto& i) {
               return not contains({"@literal", "@param", "@return", "convolution", "add", "relu"},
                                   i.name());
           }))
            return;
        std::sort(names.begin(), names.end());
        module_ref mm = mpm.create_module("mlir_" + pm->name());
        mm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> param_map;
        auto x    = mm->add_parameter("x" + std::to_string(names.size()),
                                  conv_ins->inputs().at(0)->get_shape());
        auto w    = mm->add_parameter("x" + std::to_string(names.size() + 1),
                                  conv_ins->inputs().at(1)->get_shape());
        auto conv = mm->add_instruction(conv_ins->get_operator(), {x, w});
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
        mm->add_return(mm->insert_module_instructions(mm->end(), pm, param_map));

        std::vector<instruction_ref> inputs;
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != conv_ins; });
        inputs.insert(inputs.end(), conv_ins->inputs().begin(), conv_ins->inputs().end());
        mpm.get_module().replace_instruction(
            ins, mlir_conv{conv_ins->get_operator()}, inputs, {mm});
    }
};
} // namespace

void fuse_mlir::apply(module_pass_manager& mpm) const { match::find_matches(mpm, find_conv_pointwise{}); }

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
