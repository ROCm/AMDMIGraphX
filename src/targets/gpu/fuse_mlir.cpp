#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

struct find_conv_pointwise
{
    // Find a convolution followed by a pointwise operation.
    auto matcher() const
    {
        auto convolution =
            match::skip(match::name("contiguous"))(match::name("convolution").bind("convolution"));
        return match::name("pointwise")(match::any_of[match::inputs()](convolution.bind("x")));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto ins      = r.result;
        auto conv_ins = r.instructions["convolution"];
        auto x_ins    = r.instructions["x"]; // input after contiguous
        auto pm       = ins->module_inputs().front();
        auto names    = pm->get_parameter_names();
        std::sort(names.begin(), names.end());
        module mm{};
        std::unordered_map<instruction_ref, instruction_ref> param_map;
        auto x    = mm.add_parameter("x" + std::to_string(names.size()),
                                  conv_ins->inputs().at(0)->get_shape());
        auto w    = mm.add_parameter("x" + std::to_string(names.size() + 1),
                                  conv_ins->inputs().at(1)->get_shape());
        auto conv = mm.add_instruction(conv_ins->get_operator(), {x, w});
        std::transform(names.begin(),
                       names.end(),
                       ins->inputs().begin(),
                       std::inserter(param_map, param_map.end()),
                       [&](auto name, auto input) {
                           if(input == x_ins)
                               return std::make_pair(pm->get_parameter(name), conv);
                           return std::make_pair(pm->get_parameter(name),
                                                 mm.add_parameter(name, input->get_shape()));
                       });
        mm.add_return(mm.insert_module_instructions(mm.end(), pm, param_map));

        auto inputs = ins->inputs();
        inputs.insert(inputs.end(), conv_ins->inputs().begin(), conv_ins->inputs().end());
        inputs.push_back(m.insert_instruction(
            ins, make_op("hip::allocate", {{"shape", to_value(ins->get_shape())}})));
        auto mlir = insert_mlir(m, ins, mm, inputs);
        m.replace_instruction(ins, mlir);
    }
};

void fuse_mlir::apply(module& m) const { match::find_matches(m, find_conv_pointwise{}); }

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
