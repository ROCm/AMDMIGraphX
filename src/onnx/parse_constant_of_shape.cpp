#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_constant_of_shape : op_parser<parse_constant_of_shape>
{
    std::vector<op_desc> operators() const { return {{"ConstantOfShape"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        literal l_val{};
        if(contains(info.attributes, "value"))
        {
            l_val = parser.parse_value(info.attributes.at("value"));
            if(l_val.get_shape().elements() != 1)
            {
                MIGRAPHX_THROW("ConstantOfShape: attribute value can contain only 1 elements!");
            }
        }
        else
        {
            l_val = literal({shape::float_type, {1}, {0}}, {0.0f});
        }

        // input is empty, output is a scalar
        auto type = l_val.get_shape().type();

        if(args.empty())
        {
            MIGRAPHX_THROW("ConstantOfShape : must have 1 input!");
        }
        else
        {
            migraphx::shape s;
            // empty input tensor, output is a scalar
            if(args[0]->get_shape().elements() == 0)
            {
                s = migraphx::shape{type, {1}, {0}};
            }
            else
            {
                migraphx::argument in = args[0]->eval();
                check_arg_empty(in, "ConstantOfShape: dynamic shape is not supported");

                std::vector<std::size_t> dims;
                in.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
                s = migraphx::shape{type, dims};
            }

            literal l_out{};
            l_val.visit([&](auto val) {
                using val_type = std::remove_cv_t<typename decltype(val)::value_type>;
                // l_val contains only one element
                std::vector<val_type> out_vec(s.elements(), val.front());
                l_out = literal(s, out_vec);
            });

            return info.add_literal(l_out);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
