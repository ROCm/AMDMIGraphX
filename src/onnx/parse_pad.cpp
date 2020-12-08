#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void calc_reflect_indices(std::vector<int>& indices, const int64_t num_dims)
{
    int k         = 0;
    bool reversed = false;
    // in reflect padding, if the num_pads > num_dims,
    // compute the extra pad indices periodically, ex. ( 1, 2, 3, 2, 1, 0)
    for(int& idx : indices)
    {
        if(k == num_dims - 1)
            reversed = true;
        if(k == 0)
            reversed = false;
        if(reversed)
            k--;
        else
            k++;
        idx = k;
    }
}

instruction_ref reflect_pad(const onnx_parser::node_info& info,
                            const std::vector<int64_t>& pads,
                            instruction_ref input)
{
    size_t num_dims = pads.size() / 2;
    std::vector<int> ldims(pads.begin(), pads.begin() + num_dims);
    std::vector<int> rdims(pads.begin() + num_dims, pads.end());
    assert(ldims.size() == rdims.size());

    std::vector<int64_t> axes(num_dims);
    std::iota(axes.begin(), axes.end(), int64_t{0});

    // iterate over dimensions, starting from lowest dimension
    for(int64_t i = num_dims - 1; i >= 0; i--)
    {
        auto axis   = i;
        auto lcount = ldims.at(i);
        auto rcount = rdims.at(i);
        if(lcount == 0 and rcount == 0) // no padding for current dim
            continue;

        // calculate starts and ends for each iteration since shape may change
        std::vector<size_t> dims = input->get_shape().lens();
        std::vector<int64_t> starts(axes.size(), 0);
        std::vector<int64_t> ends(dims.begin(), dims.end());
        std::vector<instruction_ref> slices;

        auto starts_it = starts.begin() + i;
        auto ends_it   = ends.begin() + i;
        auto dims_it   = dims.begin() + i;

        std::vector<int> l_indices(lcount);
        std::vector<int> r_indices(rcount);

        // compute slice indices in a periodic fashion
        calc_reflect_indices(l_indices, *dims_it);
        calc_reflect_indices(r_indices, *dims_it);

        for(int idx : l_indices)
        {
            *starts_it = idx;
            *ends_it   = *starts_it + 1;
            slices.push_back(info.mm->add_instruction(
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), input));
        }
        // when padding on the left side, the outermost pad should be at the beginning
        std::reverse(slices.begin(), slices.end());
        slices.push_back(input);
        for(int idx : r_indices)
        {
            *starts_it = *dims_it - idx - 1;
            *ends_it   = *starts_it + 1;
            slices.push_back(info.mm->add_instruction(
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), input));
        }
        input = info.mm->add_instruction(make_op("concat", {{"axis", axis}}), slices);
    }
    return input;
}

struct parse_pad : op_parser<parse_pad>
{
    std::vector<op_desc> operators() const { return {{"Pad"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<int64_t> pads{};
        if(args.size() >= 2)
        {
            auto pad_arg = args.at(1)->eval();
            check_arg_empty(pad_arg, "PARSE_PAD: pad input must be constant");
            pad_arg.visit([&](auto v) { pads.assign(v.begin(), v.end()); });
        }
        else if(contains(info.attributes, "pads"))
        {
            auto&& pad_vals = info.attributes["pads"].ints();
            pads            = std::vector<int64_t>(pad_vals.begin(), pad_vals.end());
        }
        else
        {
            MIGRAPHX_THROW("PARSE_PAD: pad must be available");
        }

        // check if padding is actually being done (at least one value is nonzero)
        if(std::all_of(pads.begin(), pads.end(), [](const int& i) { return i == 0; }))
        {
            return info.add_instruction(make_op("identity"), args.front());
        }

        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode == "reflect")
                return reflect_pad(info, pads, args.front());
            if(mode != "constant")
            {
                MIGRAPHX_THROW(
                    "PARSE_PAD: migraphx currently only supports constant and reflect padding");
            }
        }

        float value = 0.0f;
        // third input is the value
        if(args.size() == 3)
        {
            auto val_ins = args.at(2);
            if(!val_ins->can_eval())
            {
                MIGRAPHX_THROW("PARSE_PAD: input value must be constant");
            }
            auto val_arg = val_ins->eval();
            if(val_arg.get_shape().elements() != 1)
            {
                MIGRAPHX_THROW("PARSE_PAD: value should contain only one element");
            }
            value = val_arg.at<float>();
        }
        else if(contains(info.attributes, "value"))
        {
            value = parser.parse_value(info.attributes.at("value")).at<float>();
        }

        return info.add_instruction(migraphx::make_op("pad", {{"pads", pads}, {"value", value}}),
                                    args.front());
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
