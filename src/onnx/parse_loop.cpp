#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_loop : op_parser<parse_loop>
{
    std::vector<op_desc> operators() const { return {{"Loop"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        // default value of the max_iter_num
        int64_t max_iterations = parser.max_loop_iterations;
        // iteration input is empty
        if(args.at(0)->name() == "undefined")
        {
            shape iter_s{shape::int64_type};
            args[0] = info.add_literal(literal(iter_s, {max_iterations}));
        }
        else
        {
            auto arg_iters = args.at(0)->eval();
            if(not arg_iters.empty())
            {
                max_iterations = arg_iters.at<int64_t>();
            }
        }

        // condition input is empty
        if(args.at(1)->name() == "undefined")
        {
            shape cond_s{shape::bool_type};
            args[1] = info.add_literal(literal(cond_s, {true}));
        }

        // retrieve the subgraph
        const auto& sub_graph = info.attributes.at("body").g();
        std::string mod_name  = info.name + "_loop";
        module_ref sub_mod    = parser.prog.create_module(mod_name);

        // parse the sub_graph
        parser.parse_graph(sub_mod, sub_graph);

        auto ret = info.add_instruction(
            make_op("loop", {{"max_iterations", max_iterations}}), args, {sub_mod});
        auto out_s = ret->get_shape();
        assert(out_s.type() == shape::tuple_type);

        const auto& vec_shapes = out_s.sub_shapes();
        std::vector<instruction_ref> out_inss;
        for(std::size_t i = 0; i < vec_shapes.size(); ++i)
        {
            auto r = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), ret);
            out_inss.push_back(r);
        }

        return out_inss;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
