#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_if : op_parser<parse_if>
{
    std::vector<op_desc> operators() const { return {{"If"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        migraphx::argument cond_arg = args.front()->eval();
        // cond is not constant, need to create sub_modules
        if(cond_arg.empty())
        {
            MIGRAPHX_THROW(
                "PARSE_IF: current implementation requires condition input to be constant!");
        }

        std::vector<bool> vec_conds;
        cond_arg.visit([&](auto s) { vec_conds.assign(s.begin(), s.end()); });
        if(vec_conds.size() != 1)
        {
            MIGRAPHX_THROW("PARSE_IF: condition input can have only one element!");
        }

        auto* mod = info.mod;
        // then branch
        if(vec_conds.front())
        {
            const auto& then_graph = info.attributes.at("then_branch").g();
            parser.parse_graph(mod, then_graph);
        }
        // else branch
        else
        {
            const auto& else_graph = info.attributes.at("else_branch").g();
            parser.parse_graph(mod, else_graph);
        }

        // inputs of the return instruction are that of the output of the
        // if instruction
        instruction_ref ret_ins = std::prev(mod->end());
        auto outputs            = ret_ins->inputs();
        assert(ret_ins->name() == "@return");
        mod->remove_instruction(ret_ins);

        return outputs;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
