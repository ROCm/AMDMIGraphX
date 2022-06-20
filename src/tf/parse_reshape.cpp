#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_reshape : op_parser<parse_reshape>
{
    std::vector<op_desc> operators() const { return {{"Reshape"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(args.size() != 2)
            MIGRAPHX_THROW("reshape needs 2 arguments (input, new_shape)");
        auto s = args[1]->eval();
        std::vector<int64_t> dims;
        s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
        return info.add_instruction(make_op("reshape", {{"dims", dims}}),
                                    info.make_contiguous(args[0]));
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
