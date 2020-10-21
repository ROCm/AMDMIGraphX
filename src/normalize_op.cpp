#include <migraphx/operation.hpp>
#include <migraphx/normalize_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void normalize_op(operation& op, std::vector<shape> inputs)
{
    int64_t n_dim = static_cast<int64_t>(inputs[0].lens().size());
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
        if(std::any_of(axes.begin(), axes.end(), [=](auto i) { return i < 0; }))
        {
            std::transform(axes.begin(), axes.end(), axes.begin(), [&](auto i) {
                return ((i < 0) ? i + n_dim : i);
            });
            val["axes"] = axes;
            op.from_value(val);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

