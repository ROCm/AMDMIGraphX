#include <migraphx/operation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/normalize_attribute.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// different attributes
// 1) use_input(default)/use_output
// 2) use_rank(default)/use_len
// 3) clip_min(default)/not_clip_min
//   3.1) include_min(default)/exclude_min
// 4) clip_max(default)/not_clip_max
//   4.1) exclude_max(default)/include_max
auto tune_attribute(const std::vector<int64_t>& vec,
                    const std::vector<int64_t>& axes,
                    const value& val,
                    const std::vector<std::size_t>& lens)
{
    std::vector<int64_t> result(vec);
    int64_t n_rank                                 = static_cast<int64_t>(lens.size());
    std::vector<op::normalize_attribute> vec_attrs = val.to_vector<op::normalize_attribute>();
    if(contains(vec_attrs, op::normalize_attribute::use_output))
    {
        n_rank = n_rank + vec.size();
    }

    std::vector<int64_t> max_vals(vec.size(), n_rank);
    if(contains(vec_attrs, op::normalize_attribute::use_len))
    {
        std::transform(axes.begin(), axes.end(), max_vals.begin(), [&](auto i) { return lens[i]; });
    }

    if(contains(vec_attrs, op::normalize_attribute::clip_max))
    {
        if(contains(vec_attrs, op::normalize_attribute::include_max))
        {
            std::transform(result.begin(),
                           result.end(),
                           max_vals.begin(),
                           result.begin(),
                           [](auto v, auto mv) { return v > mv ? mv : v; });
        }
        else
        {
            std::transform(result.begin(),
                           result.end(),
                           max_vals.begin(),
                           result.begin(),
                           [](auto v, auto mv) { return v >= mv ? mv - 1 : v; });
        }
    }
    else
    {
        if(contains(vec_attrs, op::normalize_attribute::include_max))
        {
            if(!std::equal(result.begin(), result.end(), max_vals.begin(), std::less_equal<>{}))
            {
                MIGRAPHX_THROW("TUNE_VECTOR: value out of range!");
            }
        }
        else
        {
            if(!std::equal(result.begin(), result.end(), max_vals.begin(), std::less<>{}))
            {
                MIGRAPHX_THROW("TUNE_VECTOR: value out of range!");
            }
        }
    }

    std::vector<int64_t> min_vals = max_vals;
    std::transform(min_vals.begin(), min_vals.end(), min_vals.begin(), [](auto v) { return -v; });
    if(contains(vec_attrs, op::normalize_attribute::clip_min))
    {
        if(contains(vec_attrs, op::normalize_attribute::include_min))
        {
            std::transform(result.begin(),
                           result.end(),
                           min_vals.begin(),
                           result.begin(),
                           [](auto v, auto mv) { return v < mv ? mv : v; });
        }
        else
        {
            std::transform(result.begin(),
                           result.end(),
                           min_vals.begin(),
                           result.begin(),
                           [](auto v, auto mv) { return v < mv + 1 ? mv + 1 : v; });
        }
    }
    else
    {
        if(contains(vec_attrs, op::normalize_attribute::include_min))
        {
            if(!std::equal(min_vals.begin(), min_vals.end(), result.begin(), std::less_equal<>{}))
            {
                MIGRAPHX_THROW("TUNE_VECTOR: attribute out of range!");
            }
        }
        else
        {
            if(!std::equal(result.begin(), result.end(), min_vals.begin(), std::less<>{}))
            {
                MIGRAPHX_THROW("TUNE_VECTOR: attribute out of range!");
            }
        }
    }

    std::transform(
        result.begin(), result.end(), max_vals.begin(), result.begin(), [](auto v, auto mv) {
            return v < 0 ? v + mv : v;
        });

    return result;
}

bool normalize_attributes(operation& op, const std::vector<std::size_t>& lens)
{
    bool tuned = false;
    auto attrs = op.attributes();
    auto val   = op.to_value();
    if(!attrs.contains("normalize_axes"))
    {
        return false;
    }

    auto attr_v = attrs.at("normalize_axes").without_key();
    for(const auto& rv : attr_v)
    {
        const auto& key = rv.get_key();
        if(val.contains(key))
        {
            auto vv = val.at(key).without_key();
            if(vv.is_array())
            {
                std::vector<int64_t> axes;
                if(val.contains("axes"))
                {
                    axes = val.at("axes").without_key().to_vector<int64_t>();
                }
                auto vec    = vv.to_vector<int64_t>();
                auto result = tune_attribute(vec, axes, rv.without_key(), lens);
                val[key]    = result;
                op.from_value(val);
                val   = op.to_value();
                tuned = true;
            }
            else
            {
                auto num    = vv.to<int64_t>();
                auto result = tune_attribute({num}, {num}, rv.without_key(), lens);
                val[key]    = result.front();
                op.from_value(val);
                val   = op.to_value();
                tuned = true;
            }
        }
        else
        {
            MIGRAPHX_THROW("NORMALIZE_ATTR : op " + op.name() + " attribute \"" + key +
                           "\" not exist!");
        }
    }

    return tuned;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
