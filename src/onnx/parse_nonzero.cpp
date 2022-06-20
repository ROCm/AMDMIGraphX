#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

template <class T>
static std::vector<std::size_t> nonzero_indices(const std::vector<T>& data)
{
    std::vector<std::size_t> indices;
    for(std::size_t i = 0; i < data.size(); ++i)
    {
        if(!float_equal(data[i], 0))
            indices.push_back(i);
    }

    return indices;
}

struct parse_nonzero : op_parser<parse_nonzero>
{
    std::vector<op_desc> operators() const { return {{"NonZero"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        migraphx::argument data_arg = args.back()->eval();
        if(data_arg.empty())
        {
            return info.add_instruction(make_op("nonzero"), args);
        }
        else
        {
            std::vector<std::size_t> indices;
            data_arg.visit([&](auto val) {
                using val_type = std::remove_cv_t<typename decltype(val)::value_type>;
                std::vector<val_type> vec_data;
                vec_data.assign(val.begin(), val.end());
                indices = nonzero_indices(vec_data);
            });

            shape in_s = args[0]->get_shape();
            shape out_s{shape::int64_type, {in_s.lens().size(), indices.size()}};

            std::vector<int64_t> out_data(out_s.elements());
            for(std::size_t i = 0; i < indices.size(); ++i)
            {
                auto idx = in_s.multi(indices[i]);
                for(std::size_t j = 0; j < in_s.lens().size(); ++j)
                {
                    out_data[out_s.index({j, i})] = idx[j];
                }
            }

            return info.add_literal(literal(out_s, out_data));
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
