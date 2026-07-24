/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/operation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/op/common.hpp>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// min/max that fold to one operand when the ordering is provable via intervals,
// and only fall back to a symbolic min/max node when it is indeterminate.
static sym::expr fold_min(const sym::expr& a, const sym::expr& b)
{
    auto lt = sym::strict_less(a, b);
    if(lt.has_value())
        return *lt ? a : b;
    return sym::min(a, b);
}
static sym::expr fold_max(const sym::expr& a, const sym::expr& b)
{
    auto lt = sym::strict_less(a, b);
    if(lt.has_value())
        return *lt ? b : a;
    return sym::max(a, b);
}

static sym::expr axis_len_expr(const shape& s, int64_t axis)
{
    if(not s.dynamic())
        return sym::lit(static_cast<int64_t>(s.lens().at(axis)));
    const auto& dd = s.dyn_dims().at(axis);
    if(dd.is_symbolic())
        return dd.sym_expr;
    if(dd.is_fixed())
        return sym::lit(static_cast<int64_t>(dd.get_interval().max));
    MIGRAPHX_THROW("normalize_attributes: cannot normalize a symbolic bound on a non-fixed axis");
}

static dim_like to_dim_like(const sym::expr& e)
{
    if(e.name() == "literal")
        return sym::to<int64_t>(e.eval({}));
    return shape::dynamic_dimension{e};
}

// Symbolic analog of tune_attribute for dim_like bounds. Requires use_len normalization
// (throws otherwise) and applies the ONNX clamp norm(v) = clamp(v < 0 ? v + D : v, 0, D)
// symbolically, folding against the interval bounds where provable.
template <class Message>
static value tune_attribute_sym(const std::vector<dim_like>& dims,
                                const std::vector<int64_t>& axes,
                                const std::vector<op::normalize_attribute>& attrs,
                                const shape& input_shape,
                                Message m)
{
    if(not contains(attrs, op::normalize_attribute::use_len))
        MIGRAPHX_THROW(m() + "symbolic bounds are only supported with use_len normalization");
    if(axes.size() != dims.size())
        MIGRAPHX_THROW(m() + "symbolic bounds require one axis per bound");
    auto zero  = sym::lit(std::int64_t{0});
    auto exprs = to_sym_exprs(dims);
    std::vector<dim_like> result(dims.size());
    std::transform(
        exprs.begin(), exprs.end(), axes.begin(), result.begin(), [&](const auto& v, auto axis) {
            auto len = axis_len_expr(input_shape, axis);
            auto neg = sym::strict_less(v, zero); // from-the-end (negative) index?
            if(not neg.has_value())
                MIGRAPHX_THROW(m() + "symbolic bound of indeterminate sign cannot be normalized");
            auto abs_v = *neg ? v + len : v;
            return to_dim_like(fold_min(fold_max(abs_v, zero), len));
        });
    return migraphx::to_value(result);
}

/**
 * Parameters:
 * vec: the vector attribute to normalize
 * axes: the operator's axes attribute if it exists, empty otherwise
 * val: the normalize_axes key and options. Ex: normalize["axes"] =
 * value::array{normalize_attribute::include_min};
 * input_shape: input shape passed when calling
 * normalize_attributes(op&, input_shape)
 *
 * See normalize_attribute.hpp for explaining the options.
 */
template <class Message>
static auto tune_attribute(const std::vector<int64_t>& vec,
                           const std::vector<int64_t>& axes,
                           const value& val,
                           const shape& input_shape,
                           Message m)
{
    std::vector<int64_t> result(vec);
    if(result.empty())
    {
        return result;
    };
    int64_t n_rank                                 = input_shape.ndim();
    std::vector<op::normalize_attribute> vec_attrs = val.to_vector<op::normalize_attribute>();
    if(contains(vec_attrs, op::normalize_attribute::use_output))
    {
        n_rank = n_rank + vec.size();
    }

    std::vector<int64_t> max_vals(vec.size(), n_rank);

    if(contains(vec_attrs, op::normalize_attribute::use_len))
    {
        if(input_shape.dynamic())
        {
            // return the unchanged `vec` if the dynamic_dimensions at `axes` are not fixed
            if(std::any_of(axes.begin(), axes.end(), [&](auto ax) {
                   return not input_shape.dyn_dims().at(ax).is_fixed();
               }))
            {
                return vec;
            }
            std::transform(axes.begin(), axes.end(), max_vals.begin(), [&](auto i) {
                return input_shape.dyn_dims().at(i).get_interval().max;
            });
        }
        else
        {
            std::transform(axes.begin(), axes.end(), max_vals.begin(), [&](auto i) {
                return input_shape.lens().at(i);
            });
        }
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
            if(not std::equal(result.begin(), result.end(), max_vals.begin(), std::less_equal<>{}))
            {
                MIGRAPHX_THROW(m() + "value out of range!");
            }
        }
        else
        {
            if(not std::equal(result.begin(), result.end(), max_vals.begin(), std::less<>{}))
            {
                MIGRAPHX_THROW(m() + "value out of range!");
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
            if(not std::equal(
                   min_vals.begin(), min_vals.end(), result.begin(), std::less_equal<>{}))
            {
                MIGRAPHX_THROW(m() + "attribute out of range!");
            }
        }
        else
        {
            if(not std::equal(result.begin(), result.end(), min_vals.begin(), std::less<>{}))
            {
                MIGRAPHX_THROW(m() + "attribute out of range!");
            }
        }
    }

    std::transform(
        result.begin(), result.end(), max_vals.begin(), result.begin(), [](auto v, auto mv) {
            return v < 0 ? v + mv : v;
        });

    return result;
}

static auto tune_pad_attribute(const value& val)
{

    std::vector<size_t> vec_attrs = val.to_vector<size_t>();
    std::vector<size_t> result(vec_attrs.begin(), vec_attrs.end());
    std::copy(vec_attrs.begin(), vec_attrs.end(), std::back_inserter(result));

    return result;
}

/**
 * Assumptions:
 *  Dimensions to pad start from the third dimension (index 2).
 *  Called by compute_shape_op() with the shape of the first input.
 */
bool normalize_attributes(operation& op, const shape& input_shape)
{
    bool tuned = false;
    auto attrs = op.attributes();
    auto val   = op.to_value();
    if(attrs.contains("normalize_padding"))
    {
        bool use_auto_padding =
            (val.contains("padding_mode") and
             (val.at("padding_mode").to<int>() != migraphx::op::padding_mode_t::default_));
        if(not use_auto_padding)
        {
            auto padding       = val.at(attrs.at("normalize_padding").to<std::string>());
            auto padding_size  = padding.size();
            auto padding_start = 2;
            if(padding_size == 2 * (input_shape.ndim() - padding_start))
                tuned = true;
            else if(padding_size != (input_shape.ndim() - padding_start))
            {
                MIGRAPHX_THROW("normalize_attributes: inconsistent padding vector size ");
            }
            else
            {
                auto result    = tune_pad_attribute(padding);
                val["padding"] = result;
                op.from_value(val);
                tuned = true;
            }
        }
    }
    if(not attrs.contains("normalize_axes"))
    {
        return tuned;
    }

    auto attr_v = attrs.at("normalize_axes").without_key();
    for(const auto& rv : attr_v)
    {
        const auto& key = rv.get_key();
        if(val.contains(key))
        {
            auto message = [&] { return op.name() + ": " + key + ": "; };
            auto vv      = val.at(key).without_key();
            if(vv.is_array())
            {
                std::vector<int64_t> axes;
                if(val.contains("axes"))
                {
                    axes = val.at("axes").without_key().to_vector<int64_t>();
                }
                // Symbolic (dim_like) bounds serialize as objects; clamp them
                // symbolically rather than against a compile-time length.
                if(std::any_of(vv.begin(), vv.end(), [](const auto& e) { return e.is_object(); }))
                {
                    auto dims = migraphx::from_value<std::vector<dim_like>>(vv);
                    val[key] =
                        tune_attribute_sym(dims,
                                           axes,
                                           rv.without_key().to_vector<op::normalize_attribute>(),
                                           input_shape,
                                           message);
                    op.from_value(val);
                    val   = op.to_value();
                    tuned = true;
                    continue;
                }
                auto vec    = vv.to_vector<int64_t>();
                auto result = tune_attribute(vec, axes, rv.without_key(), input_shape, message);
                val[key]    = result;
                op.from_value(val);
                val   = op.to_value();
                tuned = true;
            }
            else
            {
                auto num    = vv.to<int64_t>();
                auto result = tune_attribute({num}, {num}, rv.without_key(), input_shape, message);
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

std::vector<int64_t> normalize_axes(const std::vector<int64_t>& axes,
                                    const shape& input_shape,
                                    const value& attr_val,
                                    const std::string& prefix)
{
    return tune_attribute(axes, {}, attr_val, input_shape, [&] { return prefix; });
}

std::vector<int64_t> normalize_indices(const std::vector<int64_t>& indices,
                                       const std::vector<int64_t>& axes,
                                       const shape& input_shape,
                                       const value& attr_val,
                                       const std::string& prefix)
{
    return tune_attribute(indices, axes, attr_val, input_shape, [&] { return prefix; });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
