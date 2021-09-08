#include <migraphx/common.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Example:
// s0 = (3,2,4,5) and s1 = (2,1,1)
//
// In this case we need to broadcast (:,1,1) portion of
// s1 plus broadcast the 1st dimension of s1
// giving output_lens = (3,2,4,5)
//
// Another example:
// s0 = (3,2,1,5) and s1 = (2,7,5)
// In this case we need to broadcast the (:,:,1:,:) axis
// of s0 plus the 1st dimension of s1 giving
// output_lens = (3,2,7,5)
std::vector<std::size_t> compute_broadcasted_lens(std::vector<std::size_t> s0,
                                                  std::vector<std::size_t> s1)
{
    if(s0 == s1)
        return s0;
    if(s0.size() > s1.size())
        s0.swap(s1);

    std::vector<std::size_t> out_lens(s1);
    auto offset = s1.size() - s0.size();
    std::transform(
        s0.begin(), s0.end(), s1.begin() + offset, out_lens.begin() + offset, [&](auto a, auto b) {
            if(a != b and a != 1 and b != 1)
            {
                MIGRAPHX_THROW("COMPUTE_BROADCASTLEN: shape {" + to_string_range(s0) + "} and {" +
                               to_string_range(s1) + "} mismatch!");
            }
            return std::max(a, b);
        });

    return out_lens;
}

std::vector<std::size_t> compute_common_lens(const std::vector<shape>& shapes)
{
    assert(not shapes.empty());
    return transform_accumulate(shapes.begin() + 1,
                                shapes.end(),
                                shapes.front().lens(),
                                &compute_broadcasted_lens,
                                [](auto s) { return s.lens(); });
}

shape::type_t compute_common_type(shape::type_t t1, shape::type_t t2)
{
    if(t1 == t2)
        return t1;
    shape::type_t result;
    shape::visit(t1, [&](auto x) {
        shape::visit(t2, [&](auto y) {
            // Workaround broken warning on gcc 5
            (void)x;
            (void)y;
            using type = std::common_type_t<decltype(x()), decltype(y())>;
            result     = shape::get_type<type>{};
        });
    });
    return result;
}

shape::type_t compute_common_types(const std::vector<shape>& shapes)
{
    assert(not shapes.empty());
    return transform_accumulate(
        shapes.begin() + 1, shapes.end(), shapes.front().type(), &compute_common_type, [&](auto s) {
            return s.type();
        });
}

shape common_shape(const std::vector<shape>& shapes)
{
    if(shapes.empty())
        return {};
    return {compute_common_types(shapes), compute_common_lens(shapes)};
}

instruction_ref insert_common_op(module& m,
                                 instruction_ref ins,
                                 const operation& op,
                                 std::vector<instruction_ref> inputs)
{
    auto common = common_shape(to_shapes(inputs));
    std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
        if(input->get_shape().lens() != common.lens())
        {
            input = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", common.lens()}}), input);
        }
        if(input->get_shape().type() != common.type())
        {
            input = m.insert_instruction(
                ins, make_op("convert", {{"target_type", common.type()}}), input);
        }
        return input;
    });
    return m.insert_instruction(ins, op, inputs);
}

instruction_ref add_common_op(module& m, const operation& op, std::vector<instruction_ref> inputs)
{
    return insert_common_op(m, m.end(), op, std::move(inputs));
}

instruction_ref
dot_apply_alpha_beta(module& m, const std::vector<instruction_ref>& args, float alpha, float beta)
{
    auto l1       = args[0];
    auto l2       = args[1];
    auto dot_type = l1->get_shape().type();
    if(alpha != 1.0f)
    {
        auto alpha_literal = m.add_literal(alpha);
        l1                 = add_common_op(m, migraphx::make_op("mul"), {alpha_literal, l1});
        if(l1->get_shape().type() != dot_type)
        {
            l1 = m.add_instruction(make_op("convert", {{"target_type", dot_type}}), l1);
        }
    }
    auto dot_res =
        m.add_instruction(migraphx::make_op("dot"), l1, l2);
    if(args.size() == 3)
    {
        if(not float_equal(beta, 0.0f) && args[2]->get_shape().elements() > 0)
        {
            auto out_lens   = l1->get_shape().lens();
            out_lens.back() = l2->get_shape().lens().back();
            auto l3         = args[2];
            auto l3_lens    = l3->get_shape().lens();
            if(!std::equal(out_lens.begin(), out_lens.end(), l3_lens.begin(), l3_lens.end()))
            {
                l3 = m.add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), args[2]);
            }
            auto beta_literal = m.add_literal(beta);
            auto beta_l3      = add_common_op(m, migraphx::make_op("mul"), {l3, beta_literal});
            if(beta_l3->get_shape().type() != dot_type)
            {
                beta_l3 = m.add_instruction(
                    migraphx::make_op("convert", {{"target_type", dot_type}}), beta_l3);
            }
            return m.add_instruction(migraphx::make_op("add"), dot_res, beta_l3);
        }
    }
    return dot_res;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
