#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/generate.hpp>



namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_slice : op_parser<parse_slice>
{
    std::vector<op_desc> operators() const { return {{"Slice"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        op::slice op;

        // slice can have up to 5 inputs, we first check the 5th one
        // to decide whether MIGRAPHX can handle this slice
        if(args.size() == 5)
        {
            migraphx::argument step_arg = args.back()->eval();
            check_arg_empty(step_arg, "PARSE_SLICE: cannot handle variable steps for slice");
            std::vector<int> steps;
            step_arg.visit([&](auto s) { steps.assign(s.begin(), s.end()); });
            
            if(!std::all_of(steps.begin(), steps.end(), [](auto s) { return s == 1; }))
            {
                //MIGRAPHX_THROW("PARSE_SLICE: cannot handle step other than 1");

                //data
                auto arg_data = args[0];
                //data size
                auto data_s = arg_data->get_shape();
                //num of elements
                int64_t data_elem_num = static_cast<uint64_t>(data_s.lens()[0]);

                std::vector<int> data_indices(data_s.lens()[0]);
                std::generate(data_indices.begin(), data_indices.end(), [&data_elem_num]{ return --data_elem_num;});

                migraphx::shape new_shape = migraphx::shape(migraphx::shape::int64_type, {data_indices.size()});

                auto ind = info.add_literal(literal(new_shape, data_indices.begin(), data_indices.end()));

                auto op_slice_gather = make_op("gather", {{"axis", 0}});
                return info.add_instruction(op_slice_gather, arg_data, ind );
            }
        }

        if(args.size() >= 4)
        {
            migraphx::argument axes_arg = args.at(3)->eval();
            check_arg_empty(axes_arg, "PARSE_SLICE: cannot handle variable axes for slice");
            axes_arg.visit([&](auto s) { op.axes.assign(s.begin(), s.end()); });
        }
        else if(contains(info.attributes, "axes"))
        {
            literal s = parser.parse_value(info.attributes.at("axes"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.axes)); });
        }

        if(args.size() >= 3)
        {
            migraphx::argument end_arg = args.at(2)->eval();
            check_arg_empty(end_arg, "PARSE_SLICE: cannot handle variable ends for slice");
            end_arg.visit([&](auto s) { op.ends.assign(s.begin(), s.end()); });
        }
        else if(contains(info.attributes, "ends"))
        {
            literal s = parser.parse_value(info.attributes.at("ends"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.ends)); });
        }

        if(args.size() >= 2)
        {
            migraphx::argument start_arg = args.at(1)->eval();
            check_arg_empty(start_arg, "PARSE_SLICE: cannot handle variable starts for slice");
            start_arg.visit([&](auto s) { op.starts.assign(s.begin(), s.end()); });
        }
        else if(contains(info.attributes, "starts"))
        {
            literal s = parser.parse_value(info.attributes.at("starts"));
            s.visit([&](auto v) { copy(v, std::back_inserter(op.starts)); });
        }

        if(op.axes.empty())
        {
            std::vector<int64_t> axes(args[0]->get_shape().lens().size());
            std::iota(axes.begin(), axes.end(), int64_t{0});
            op.axes = axes;
        }

        return info.add_instruction(op, args[0]);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
