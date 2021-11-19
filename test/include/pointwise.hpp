#ifndef MIGRAPHX_GUARD_TEST_INCLUDE_POINTWISE_HPP
#define MIGRAPHX_GUARD_TEST_INCLUDE_POINTWISE_HPP

#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>

template <class F>
migraphx::instruction_ref add_pointwise(migraphx::program& p,
                                        const std::string& name,
                                        std::vector<migraphx::instruction_ref> inputs,
                                        F f)
{
    auto* pm = p.create_module(name);
    auto* mm = p.get_main_module();
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto input) {
        return pm->add_parameter("x" + std::to_string(params.size()),
                                 migraphx::shape{input->get_shape().type()});
    });
    auto r = f(pm, params);
    pm->add_return({r});
    return mm->add_instruction(migraphx::make_op("pointwise"), inputs, {pm});
}

inline auto single_pointwise(const std::string& name)
{
    return [=](auto* pm, const auto& inputs) {
        return pm->add_instruction(migraphx::make_op(name), inputs);
    };
}

#endif // MIGRAPHX_GUARD_TEST_INCLUDE_POINTWISE_HPP
