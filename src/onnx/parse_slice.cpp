#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

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

        bool reverse_direction = false;

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
                if(std::all_of(steps.begin(), steps.end(), [](auto s) { return s == -1; }))
                {
                    reverse_direction = true;
                }
                else
                {
                    MIGRAPHX_THROW("PARSE_SLICE: cannot handle step other than 1 or -1");
                }
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

        if(reverse_direction == true){
            //auto tmp = op.starts;
            //op.starts = op.ends;
            //op.ends = tmp;

            migraphx::argument axes_arg = args.at(3)->eval();
            std::vector<int> axes_v;
            axes_arg.visit([&](auto s) { axes_v.assign(s.begin(), s.end()); });

            auto lens = args[0]->get_shape().lens();
            
            for(auto axis: axes_v){
                auto start_v   = op.starts[axis];
                auto end_v     = op.ends[axis];
                std::cout << "PARSE debug: starts:" << start_v << " ends:" << end_v << std::endl;
                std::cout << "PARSE debug: INT_MIN:" << INT_MIN << std::endl;
                std::cout << "PARSE debug: LENS[AXIS]:" << lens[axis] << std::endl;
                if ( (start_v < 0) & (end_v < INT_MIN)) {
                    std::cout << "ok";
                    op.ends[axis]      = lens[axis] + start_v + 1;
                    op.starts[axis]    = 0;
                }
                else if ( (start_v < 0) & (end_v > INT_MIN) & (end_v < 0)) {
                    op.ends[axis]      = lens[axis] + start_v + 1;
                    op.starts[axis]    = end_v - INT_MIN;
                }
            }
            
            std::cout << "CHECK starts:";
            for (auto k: op.starts){
                std::cout << k << " " << std::endl;
            }

            std::cout << "CHECK ends:";
            for (auto k: op.ends){
                std::cout << k << " " << std::endl;
            }

            
            auto ll1 = info.add_instruction(op, args[0]); 
            auto ll2 = info.add_instruction(make_op("reverse", {{"axis",0}}), ll1); //TODO: take care of axis here
            return ll2;
        } else {
            return info.add_instruction(op, args[0]);
        }

    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
