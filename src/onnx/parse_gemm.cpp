#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gemm : op_parser<parse_gemm>
{
    std::vector<op_desc> operators() const { return {{"Gemm"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        float alpha = 1.0f;
        float beta  = 1.0f;
        bool transa = false;
        bool transb = false;
        if(contains(info.attributes, "alpha"))
        {
            alpha = parser.parse_value(info.attributes.at("alpha")).at<float>();
        }
        if(contains(info.attributes, "beta"))
        {
            beta = parser.parse_value(info.attributes.at("beta")).at<float>();
        }
        if(contains(info.attributes, "transA"))
        {
            transa = parser.parse_value(info.attributes.at("transA")).at<bool>();
        }
        if(contains(info.attributes, "transB"))
        {
            transb = parser.parse_value(info.attributes.at("transB")).at<bool>();
        }

        std::vector<int64_t> perm(args[0]->get_shape().lens().size());
        std::iota(perm.begin(), perm.end(), int64_t{0});
        // swap the last two elements
        std::swap(*perm.rbegin(), *(perm.rbegin() + 1));

        auto l1 = args[0];

        if(alpha != 1.0f)
        {
            auto alpha_literal = info.add_literal(alpha);
            auto alpha_l1      = info.add_broadcastable_binary_op("mul", alpha_literal, l1);
            l1 = info.add_instruction(make_op("convert", {{"target_type", l1->get_shape().type()}}),
                                      alpha_l1);
        }

        l1      = (transa) ? info.add_instruction(make_op("transpose", {{"dims", perm}}), l1) : l1;
        auto l2 = (transb) ? info.add_instruction(make_op("transpose", {{"dims", perm}}), args[1])
                           : args[1];

        if(args.size() == 3)
        {
            if(beta != 0.0f && args[2]->get_shape().elements() > 0)
            {
                auto out_lens   = l1->get_shape().lens();
                out_lens.back() = l2->get_shape().lens().back();
                auto l3         = args[2];
                auto l3_lens    = l3->get_shape().lens();
                if(!std::equal(out_lens.begin(), out_lens.end(), l3_lens.begin(), l3_lens.end()))
                {
                    l3 = info.add_instruction(
                        make_op("multibroadcast", {{"output_lens", out_lens}}), args[2]);
                }
                auto beta_literal   = info.add_literal(beta);
                auto beta_broadcast = info.add_instruction(
                    make_op("multibroadcast", {{"output_lens", out_lens}}), beta_literal);
                l3 = info.add_instruction(make_op("mul"), l3, beta_broadcast);

                return info.add_instruction(
                    make_op("dot", {{"alpha", 1.0f}, {"beta", 1.0f}}), l1, l2, l3);
            }
        }

        return info.add_instruction(make_op("dot", {{"alpha", 1.0f}, {"beta", 1.0f}}), l1, l2);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
