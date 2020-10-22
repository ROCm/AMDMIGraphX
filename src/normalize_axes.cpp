#include <migraphx/operation.hpp>
#include <migraphx/normalize_axes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void normalize_axes(operation& op, const std::vector<std::size_t>& lens)
{
    int64_t n_dim = static_cast<int64_t>(lens.size());
    value val     = op.to_value();
    if(val.contains("axis"))
    {
        auto axis = val["axis"].without_key().to<int64_t>();
        if(axis < 0)
        {
            axis        = axis < 0 ? axis + n_dim : axis;
            val["axis"] = axis;
            op.from_value(val);
        }
    }
    else if(val.contains("axes"))
    {
        auto axes = val["axes"].without_key().to_vector<int64_t>();

        if(op.name() == "unsqueeze")
        {
            n_dim = n_dim + axes.size();
        }

        bool tuned = false;
        if(std::any_of(axes.begin(), axes.end(), [=](auto i) { return i < 0; }))
        {
            std::transform(axes.begin(), axes.end(), axes.begin(), [&](auto i) {
                return ((i < 0) ? i + n_dim : i);
            });
            val["axes"] = axes;
            tuned       = true;
        }

        // for slice
        std::vector<int64_t> axis_lens(axes.size());
        std::transform(
            axes.begin(), axes.end(), axis_lens.begin(), [&](auto axis) { return lens[axis]; });

        if(val.contains("starts"))
        {
            auto starts = val["starts"].without_key().to_vector<int64_t>();
            if(std::any_of(starts.begin(), starts.end(), [&](auto i) { return i < 0; }) or
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
                val["starts"] = starts;
                tuned         = true;
            }
        }

        if(val.contains("ends"))
        {
            auto ends = val["ends"].without_key().to_vector<int64_t>();
            if(std::any_of(ends.begin(), ends.end(), [&](auto i) { return i < 0; }) or
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
                val["ends"] = ends;
                tuned       = true;
            }
        }

        if(tuned)
        {
            op.from_value(val);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
