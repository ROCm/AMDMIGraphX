#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_matmul : op_parser<parse_matmul>
{
    std::vector<op_desc> operators() const
    {
        return {{"BatchMatMul"}, {"BatchMatMulV2"}, {"MatMul"}};
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        bool transa = false;
        bool transb = false;

        if(contains(info.attributes, "transpose_a"))
        {
            transa = info.attributes.at("transpose_a").b();
        }
        if(contains(info.attributes, "transpose_b"))
        {
            transb = info.attributes.at("transpose_b").b();
        }

        if(contains(info.attributes, "adj_x"))
        {
            transa = info.attributes.at("adj_x").b();
        }
        if(contains(info.attributes, "adj_y"))
        {
            transb = info.attributes.at("adj_y").b();
        }

        std::vector<int64_t> perm(args[0]->get_shape().lens().size());
        std::iota(perm.begin(), perm.end(), int64_t{0});
        // swap the last two elements
        std::iter_swap(perm.end() - 1, perm.end() - 2);

        auto l1 = (transa)
                      ? info.add_instruction(make_op("transpose", {{"permutation", perm}}), args[0])
                      : args[0];
        auto l2 = (transb)
                      ? info.add_instruction(make_op("transpose", {{"permutation", perm}}), args[1])
                      : args[1];

        return info.add_instruction(make_op("dot"), l1, l2);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
