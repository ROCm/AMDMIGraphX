#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

const auto& get_nearest_op(const std::string& mode)
{
    using nearest_op = std::function<std::size_t(std::size_t, double)>;
    static std::unordered_map<std::string, nearest_op> const nearest_ops = {
        {"round_prefer_floor",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::ceil((val - 0.5)));
         }},
        {"round_prefer_ceil",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::round((val)));
         }},
        {"floor",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::floor((val)));
         }},
        {"ceil", [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::ceil((val)));
         }}};

    if(!contains(nearest_ops, mode))
    {
        MIGRAPHX_THROW("PARSE_RESIZE: nearest_mode " + mode + " not supported!");
    }

    return nearest_ops.at(mode);
}

const auto& get_original_idx_op(const std::string& mode)
{
    using original_idx_op = std::function<double(std::size_t, std::size_t, std::size_t, double)>;
    static std::unordered_map<std::string, original_idx_op> const idx_ops = {
        {"half_pixel",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale - 0.5;
         }},
        {"pytorch_half_pixel",
         [=](std::size_t, std::size_t l_out, std::size_t idx, double scale) {
             return l_out > 1 ? (idx + 0.5) / scale - 0.5 : 0.0;
         }},
        {"align_corners",
         [=](std::size_t l_in, std::size_t l_out, std::size_t idx, double) {
             return 1.0 * idx * (l_in - 1.0) / (l_out - 1.0);
         }},
        {"asymmetric",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) { return idx / scale; }},
        {"tf_half_pixel_for_nn", [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale;
         }}};

    if(!contains(idx_ops, mode))
    {
        MIGRAPHX_THROW("PARSE_RESIZE: coordinate_transformation_mode " + mode + " not supported!");
    }

    return idx_ops.at(mode);
}

struct parse_resize : op_parser<parse_resize>
{
    std::vector<op_desc> operators() const { return {{"Resize"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::string coord_trans_mode = "half_pixel";
        if(contains(info.attributes, "coordinate_transformation_mode"))
        {
            coord_trans_mode = info.attributes.at("coordinate_transformation_mode").s();
            // does not support transformation mode "tf_crop_and_resize"
            if(coord_trans_mode == "tf_crop_and_resize")
            {
                MIGRAPHX_THROW("PARSE_RESIZE: \"tf_crop_and_resize\" mode is not supported!");
            }
        }

        // mode: only nearest mode is supported for now
        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode != "nearest")
            {
                MIGRAPHX_THROW("PARSE_RESIZE: only nearest mode is supported!");
            }
        }

        // nearest mode
        std::string nearest_mode = "round_prefer_floor";
        if(contains(info.attributes, "nearest_mode"))
        {
            nearest_mode = info.attributes.at("nearest_mode").s();
        }

        // check exclude_outside, only support 0
        if(contains(info.attributes, "exclude_outside"))
        {
            int exclude_outside = info.attributes.at("exclude_outside").i();
            if(exclude_outside == 1)
            {
                MIGRAPHX_THROW("PARSE_RESIZE: exclude_outside 1 is not supported!");
            }
        }

        // input data shape info
        auto in_s    = args[0]->get_shape();
        auto in_lens = in_s.lens();

        // output shape is explicitly specified
        std::vector<std::size_t> out_lens(in_lens.size());

        // scale
        std::vector<double> vec_scale;

        // output size is specified in input, so use it as output size
        if(args.size() == 4 and args.back()->name() != "undefined")
        {
            auto arg_out_s = args[3]->eval();
            check_arg_empty(arg_out_s, "PARSE_RESIZE: dynamic output size is not supported!");
            arg_out_s.visit([&](auto ol) { out_lens.assign(ol.begin(), ol.end()); });

            if(out_lens.size() != in_lens.size())
            {
                MIGRAPHX_THROW("PARSE_RESIZE: specified output size does not match input size");
            }

            // compute the scale
            vec_scale.resize(in_lens.size());
            std::transform(in_lens.begin(),
                           in_lens.end(),
                           out_lens.begin(),
                           vec_scale.begin(),
                           [](auto iss, auto oss) { return 1.0 * oss / iss; });
        }
        // need to compute the output lens from input
        else
        {
            auto arg_scale = args[2]->eval();
            check_arg_empty(arg_scale, "PARSE_RESIZE: dynamic input scale is not supported!");

            arg_scale.visit([&](auto v) { vec_scale.assign(v.begin(), v.end()); });
            if(in_lens.size() != vec_scale.size())
            {
                MIGRAPHX_THROW("PARSE_RESIZE: ranks of input and scale are different!");
            }

            std::transform(
                in_lens.begin(),
                in_lens.end(),
                vec_scale.begin(),
                out_lens.begin(),
                [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });
        }

        shape out_s{in_s.type(), out_lens};
        std::vector<int> ind(out_s.elements());

        // map out_idx to in_idx
        auto nearest_op = get_nearest_op(nearest_mode);
        auto idx_op     = get_original_idx_op(coord_trans_mode);

        shape_for_each(out_s, [&](auto idx) {
            auto in_idx = idx;
            for(auto ii = 0; ii < in_lens.size(); ++ii)
            {
                auto idx_val = idx_op(in_lens[ii], out_lens[ii], in_idx[ii], vec_scale[ii]);
                in_idx[ii]   = nearest_op(in_lens[ii], idx_val);
            }

            ind[out_s.index(idx)] = static_cast<int64_t>(in_s.index(in_idx));
        });

        // reshape input to one-dimension
        std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
        shape ind_s{shape::int32_type, out_lens};
        auto arg_cont = info.make_contiguous(args[0]);
        auto rsp      = info.add_instruction(make_op("reshape", {{"dims", rsp_lens}}), arg_cont);
        auto ins_ind  = info.add_literal(literal(ind_s, ind));
        return info.add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
