#include <unordered_set>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/auto_any_cast.hpp>
#include <migraphx/value.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// single axis ops {gather, concat, argmax, argmin, softmax, logsoftmax} [-r, r - 1]
bool normalize_ops::tune_axis(value& val, int64_t n_dim) const
{
    if (val.contains("axis"))
    {
        auto axis = val["axis"].without_key().to<int64_t>();
        if (axis < 0)
        {
            axis = axis < 0 ? axis + n_dim : axis;
            val["axis"] = axis;
            return true;
        }
    }
    else if (val.contains("axes"))
    {
        auto axes = val["axes"].without_key().to_vector<int64_t>();
        if (std::any_of(axes.begin(), axes.end(), [=](auto i) { return i < 0; }))
        {
            std::transform(axes.begin(), axes.end(), axes.begin(), [&](auto i) {
                return ((i < 0) ? i + n_dim : i);
            });
            val["axes"] = axes;

            return true;
        }
    }

    return false;
}

bool normalize_ops::tune_slice_inputs(std::vector<int64_t>& axes,
                        std::vector<int64_t>& starts,
                        std::vector<int64_t>& ends,
                        const std::vector<std::size_t>& lens) const
{
    // tune axes
    int64_t n_rank = static_cast<int64_t>(lens.size());
    bool tuned = false;
    if (std::any_of(axes.begin(), axes.end(), [](auto i) { return i < 0; }))
    {
        std::transform(axes.begin(), axes.end(), axes.begin(), [=](auto i) {
            return (i < 0) ? (i + n_rank) : i;
        });
        tuned = true;
    }

    std::vector<int64_t> axis_lens(axes.size());
    std::transform(axes.begin(), axes.end(), axis_lens.begin(), [&](auto axis) {
        return lens[axis];
    });

    // tune starts
    if (std::any_of(starts.begin(), starts.end(), [&] (auto i) { return i < 0; }) or 
        !std::equal(starts.begin(), starts.end(), axis_lens.begin(), std::less_equal<>{}))
    {
        std::transform(starts.begin(),
                        starts.end(),
                        axis_lens.begin(),
                        starts.begin(),
                        [=](auto i, auto dim) {
                            i = (i < -dim) ? -dim : ((i > dim) ? dim : i);
                            return (i < 0) ? (i + dim) : i;
                        });
        tuned = true;
    }

    // tune ends
    if (std::any_of(ends.begin(), ends.end(), [&] (auto i) { return i < 0; }) or 
        !std::equal(ends.begin(), ends.end(), axis_lens.begin(), std::less_equal<>{}))
    {
        std::transform(ends.begin(),
                       ends.end(),
                       axis_lens.begin(),
                       ends.begin(),
                        [=](auto i, auto dim) {
                            i = (i < -dim) ? -dim : ((i > dim) ? dim : i);
                            return (i < 0) ? (i + dim) : i;
                        });
        tuned = true;
    }

    return tuned;
}

void normalize_ops::apply(program& p) const
{
    std::unordered_set<std::string> ops_name = {"argmax", "argmin", "concat", "flatten", "gather", 
        "logsoftmax", "reduce_max", "reduce_mean", "reduce_min", "reduce_prod", "reduce_sum",
        "softmax", "squeeze", "unsqueeze"};
    for (auto ins : iterator_for(p))
    {

        if (contains(ops_name, ins->name()))
        {
            auto inputs = ins->inputs();
            int64_t n_dim = static_cast<int64_t>(inputs[0]->get_shape().lens().size());
            auto val = ins->get_operator().to_value();
            if (tune_axis(val, n_dim))
            {
                auto oper_tuned = make_op(ins->name(), val);
                p.replace_instruction(ins, oper_tuned, inputs);
            }
        }
        else if (ins->name() == "slice")
        {
            auto inputs = ins->inputs();
            auto val = ins->get_operator().to_value();
            auto axes = val["axes"].to_vector<int64_t>();
            auto sstarts = val["starts"].to_vector<int64_t>();
            auto sends = val["ends"].to_vector<int64_t>();
            auto lens = inputs[0]->get_shape().lens();

            if (tune_slice_inputs(axes, sstarts, sends, lens))
            {
                val["axes"] = axes;
                val["starts"] = sstarts;
                val["ends"] = sends;

                auto tuned_slice = make_op("slice", val);
                p.replace_instruction(ins, tuned_slice, inputs);
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
