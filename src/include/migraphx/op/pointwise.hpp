#ifndef MIGRAPHX_GUARD_OP_POINTWISE_HPP
#define MIGRAPHX_GUARD_OP_POINTWISE_HPP

#include <migraphx/config.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/module.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pointwise
{
    std::string name() const { return "pointwise"; }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
        {
            MIGRAPHX_THROW("should have one submodule.");
        }
        auto* pm    = mods.front();
        auto pnames = pm->get_parameter_names();
        std::sort(pnames.begin(), pnames.end());
        check_shapes{inputs, *this}.has(pnames.size()).same_dims();

        if(pm->get_output_shapes().size() != 1)
            MIGRAPHX_THROW("submodule should have only one output.");

        auto type = pm->get_output_shapes().front().type();

        // Scalar output if all inputs are scalar
        if(inputs.front().elements() == 1 and
           all_of(inputs, [](const auto& s) { return s.scalar(); }))
            return shape{type};

        return shape::from_permutation(type, inputs.front().lens(), find_permutation(inputs));
    }

    argument compute(const shape& output_shape,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        argument output{output_shape};
        auto* pm    = mods.front();
        auto pnames = pm->get_parameter_names();
        std::sort(pnames.begin(), pnames.end());

        par_for(output_shape.elements(), [&](auto i) {
            std::unordered_map<std::string, argument> params;

            std::transform(
                pnames.begin(),
                pnames.end(),
                args.begin(),
                std::inserter(params, params.end()),
                [&](auto&& name, auto&& arg) { return std::make_pair(name, arg.element(i)); });

            auto results = run(pm, params);
            assert(results.size() == 1);
            visit_all(output, results.front())([&](auto out, auto x) { out[i] = x.front(); });
        });
        return output;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_OP_POINTWISE_HPP
