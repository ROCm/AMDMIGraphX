#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_upsample : op_parser<parse_upsample>
{
    std::vector<op_desc> operators() const { return {{"Upsample"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode != "nearest")
            {
                MIGRAPHX_THROW("PARSE_UPSAMPLE: only nearest mode is supported!");
            }
        }

        auto arg_scale = args[1]->eval();
        check_arg_empty(arg_scale, "PARSE_UPSAMPLE: only constant scale is supported!");
        std::vector<float> vec_scale;
        arg_scale.visit([&](auto v) { vec_scale.assign(v.begin(), v.end()); });

        auto in_s    = args[0]->get_shape();
        auto in_lens = in_s.lens();
        if(in_lens.size() != vec_scale.size())
        {
            MIGRAPHX_THROW("PARSE_UPSAMPLE: ranks of input and scale are different!");
        }

        std::vector<std::size_t> out_lens(in_lens.size());
        std::transform(in_lens.begin(),
                       in_lens.end(),
                       vec_scale.begin(),
                       out_lens.begin(),
                       [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });

        std::vector<float> idx_scale(in_lens.size());
        std::transform(
            out_lens.begin(),
            out_lens.end(),
            in_lens.begin(),
            idx_scale.begin(),
            [](auto od, auto id) { return (od == id) ? 1.0f : (id - 1.0f) / (od - 1.0f); });

        shape out_s{in_s.type(), out_lens};
        std::vector<int> ind(out_s.elements());

        // map out_idx to in_idx
        shape_for_each(out_s, [&](auto idx) {
            auto in_idx = idx;
            std::transform(idx.begin(),
                           idx.end(),
                           idx_scale.begin(),
                           in_idx.begin(),
                           // nearest mode
                           [](auto index, auto scale) {
                               return static_cast<std::size_t>(std::round(index * scale));
                           });

            ind[out_s.index(idx)] = static_cast<int64_t>(in_s.index(in_idx));
        });

        // reshape input to one-dimension
        std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
        shape ind_s{shape::int32_type, out_lens};
        auto rsp     = info.add_instruction(make_op("reshape", {{"dims", rsp_lens}}), args[0]);
        auto ins_ind = info.add_literal(literal(ind_s, ind));
        return info.add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
