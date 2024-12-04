#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

#ifndef MIGRAPHX_GUARD_TEST_INCLUDE_QUANTIZE_HELPERS_HPP
#define MIGRAPHX_GUARD_TEST_INCLUDE_QUANTIZE_HELPERS_HPP

migraphx::instruction_ref broadcast_scale(migraphx::module& m,
                                          migraphx::instruction_ref scale,
                                          const std::vector<std::size_t>& out_lens,
                                          std::size_t axis)
{
    if(scale->get_shape().lens() == out_lens)
        return scale;

    migraphx::instruction_ref scale_mb;
    auto scale_lens = scale->get_shape().lens();
    if(scale_lens.front() == 1 and scale_lens.size() == 1)
        scale_mb =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), scale);
    else
        scale_mb = m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", out_lens}}), scale);
    return scale_mb;
}

migraphx::instruction_ref broadcast_shift(migraphx::module& m,
                                          migraphx::instruction_ref shift,
                                          const std::vector<std::size_t>& out_lens)
{
    if(shift->get_shape().lens() == out_lens)
        return shift;
    return m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), shift);
}

migraphx::instruction_ref add_scale_mul(migraphx::module& m,
                                        migraphx::instruction_ref scale1,
                                        migraphx::instruction_ref scale2,
                                        std::size_t axis1,
                                        std::size_t axis2,
                                        const std::vector<std::size_t>& out_lens)
{
    auto scale1_mb = broadcast_scale(m, scale1, out_lens, axis1);
    auto scale2_mb = broadcast_scale(m, scale2, out_lens, axis2);
    return m.add_instruction(migraphx::make_op("mul"), scale1_mb, scale2_mb);
}

migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                          const std::string& name,
                                          migraphx::instruction_ref x,
                                          migraphx::instruction_ref scale,
                                          migraphx::instruction_ref shift,
                                          std::size_t q_axis = 1)
{
    auto lens     = x->get_shape().lens();
    auto scale_mb = broadcast_scale(m, scale, lens, q_axis);
    auto shift_mb = broadcast_shift(m, shift, lens);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb, shift_mb);
}

migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                          const std::string& name,
                                          migraphx::instruction_ref x,
                                          migraphx::instruction_ref scale,
                                          std::size_t q_axis = 1)
{
    auto lens     = x->get_shape().lens();
    auto scale_mb = broadcast_scale(m, scale, lens, q_axis);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb);
}

#endif // MIGRAPHX_GUARD_TEST_INCLUDE_QUANTIZE_HELPERS_HPP
