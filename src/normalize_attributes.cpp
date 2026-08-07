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
#include <migraphx/optional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/op/common.hpp>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static sym::expr axis_len_expr(const shape& s, int64_t axis)
{
    if(not s.dynamic())
        return sym::lit(s.lens().at(axis));
    const auto& dd = s.dyn_dims().at(axis);
    if(dd.is_symbolic())
        return dd.sym_expr;
    if(dd.is_fixed())
        return sym::lit(dd.get_interval().max);
    MIGRAPHX_THROW("normalize_attributes: cannot normalize against a non-symbolic, non-fixed axis");
}

/**
 * Symbolic analog of tune_attribute, for attributes that hold expressions. Applies the ONNX
 * clamp norm(v) = clamp(v < 0 ? v + D : v, 0, D) symbolically, folding against the interval
 * bounds where provable.
 */
template <class Message>
static std::vector<sym::expr> tune_attribute_sym(const std::vector<sym::expr>& exprs,
                                                 const std::vector<int64_t>& axes,
                                                 const std::vector<op::normalize_attribute>& attrs,
                                                 const shape& input_shape,
                                                 Message m)
{
    if(not contains(attrs, op::normalize_attribute::use_len))
        MIGRAPHX_THROW(m() + "symbolic normalization requires use_len!");
    if(axes.size() != exprs.size())
        MIGRAPHX_THROW(m() + "one axis per value is required to normalize symbolically!");
    auto zero = sym::lit(std::int64_t{0});
    std::vector<sym::expr> result(exprs.size());
    std::transform(
        exprs.begin(), exprs.end(), axes.begin(), result.begin(), [&](const auto& v, auto axis) {
            auto len = axis_len_expr(input_shape, axis);
            auto neg = sym::strict_less(v, zero); // from-the-end (negative) index?
            if(not neg.has_value())
                MIGRAPHX_THROW(m() + "bound of indeterminate sign cannot be normalized");
            // Only a from-the-end index can land below zero once it is shifted.
            auto abs_v = *neg ? sym::fold_max(v + len, zero) : v;
            return sym::fold_min(abs_v, len);
        });
    return result;
}

/**
 * The maximum each value is normalized against: the input rank, or the length of `axes[i]` when
 * `use_len` is set. Returns nullopt when a dynamic_dimension at `axes` is not fixed, since it has
 * no single length to normalize against.
 */
template <class Message>
static optional<std::vector<int64_t>>
attribute_max_vals(std::size_t nvals,
                   const std::vector<int64_t>& axes,
                   const std::vector<op::normalize_attribute>& attrs,
                   const shape& input_shape,
                   Message m)
{
    int64_t n_rank = input_shape.ndim();
    if(contains(attrs, op::normalize_attribute::use_output))
    {
        n_rank = n_rank + nvals;
    }
    std::vector<int64_t> max_vals(nvals, n_rank);
    if(not contains(attrs, op::normalize_attribute::use_len))
        return max_vals;
    if(axes.size() > nvals)
        MIGRAPHX_THROW(m() + "more axes than values to normalize!");
    if(input_shape.dynamic() and std::any_of(axes.begin(), axes.end(), [&](auto ax) {
           return not input_shape.dyn_dims().at(ax).is_fixed();
       }))
        return nullopt;
    auto lens = input_shape.max_lens();
    std::transform(axes.begin(), axes.end(), max_vals.begin(), [&](auto i) { return lens.at(i); });
    return max_vals;
}

/**
 * Clips each value to its bound, or range checks it against the bound when clipping is off.
 * `clamp` decides which side of the bound is out of range, so the same code serves the maximum
 * and the minimum: pass std::min for an upper bound and std::max for a lower one.
 */
template <class Clamp, class Message>
static void clip_or_check(std::vector<int64_t>& result,
                          const std::vector<int64_t>& bounds,
                          bool clip,
                          Clamp clamp,
                          Message m,
                          const std::string& what)
{
    assert(result.size() == bounds.size());
    if(clip)
    {
        std::transform(result.begin(), result.end(), bounds.begin(), result.begin(), clamp);
        return;
    }
    if(not std::equal(result.begin(),
                      result.end(),
                      bounds.begin(),
                      [&](auto v, auto bound) { return v == clamp(v, bound); }))
        MIGRAPHX_THROW(m() + what);
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
static std::vector<int64_t> tune_attribute(const std::vector<int64_t>& vec,
                                           const std::vector<int64_t>& axes,
                                           const value& val,
                                           const shape& input_shape,
                                           Message m)
{
    std::vector<int64_t> result(vec);
    if(result.empty())
        return result;
    auto attrs    = val.to_vector<op::normalize_attribute>();
    auto max_vals = attribute_max_vals(vec.size(), axes, attrs, input_shape, m);
    // Without a length to normalize against, the values are returned unchanged. The caller has to
    // renormalize once the dimensions are known.
    if(not max_vals.has_value())
        return result;

    // An exclusive bound moves the limit one step inside the range.
    auto max_step = contains(attrs, op::normalize_attribute::include_max) ? 0 : 1;
    clip_or_check(result,
                  *max_vals,
                  contains(attrs, op::normalize_attribute::clip_max),
                  [&](auto v, auto bound) { return std::min(v, bound - max_step); },
                  m,
                  "value out of range!");

    // A value from the end is bounded by the negated maximum.
    std::vector<int64_t> min_vals(max_vals->size());
    std::transform(max_vals->begin(), max_vals->end(), min_vals.begin(), std::negate<>{});
    auto min_step = contains(attrs, op::normalize_attribute::include_min) ? 0 : 1;
    clip_or_check(result,
                  min_vals,
                  contains(attrs, op::normalize_attribute::clip_min),
                  [&](auto v, auto bound) { return std::max(v, bound + min_step); },
                  m,
                  "attribute out of range!");

    // Resolve the from-the-end (negative) values against the maximum.
    std::transform(
        result.begin(), result.end(), max_vals->begin(), result.begin(), [](auto v, auto mv) {
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
 * Doubles a padding attribute that only gives the padding for one side of each spatial dimension.
 * Dimensions to pad start from the third dimension (index 2). Auto padding is left to the target.
 *
 * Returns whether the padding attribute is normalized.
 */
static bool normalize_padding_attribute(operation& op,
                                        value& val,
                                        const std::string& key,
                                        const shape& input_shape)
{
    bool use_auto_padding =
        (val.contains("padding_mode") and
         (val.at("padding_mode").to<int>() != migraphx::op::padding_mode_t::default_));
    if(use_auto_padding)
        return false;
    auto padding = val.at(key);
    auto npad    = input_shape.ndim() - 2;
    if(padding.size() == 2 * npad)
        return true;
    if(padding.size() != npad)
        MIGRAPHX_THROW("normalize_attributes: inconsistent padding vector size ");
    val[key] = tune_pad_attribute(padding);
    op.from_value(val);
    return true;
}

/**
 * Normalizes an array attribute. Even a constant `sym::expr` serializes as an object, which is
 * what selects the symbolic path. See tune_attribute_sym().
 */
template <class Message>
static value tune_array_attribute(const value& vv,
                                  const std::vector<int64_t>& axes,
                                  const value& opts,
                                  const shape& input_shape,
                                  Message m)
{
    if(std::any_of(vv.begin(), vv.end(), [](const auto& e) { return e.is_object(); }))
    {
        // An expr serializes as an object carrying a "type" tag. An object without one is some
        // other attribute type, which cannot hold a normalized expression.
        if(std::any_of(vv.begin(), vv.end(), [](const auto& e) {
               return e.is_object() and not e.contains("type");
           }))
            MIGRAPHX_THROW(m() + "symbolic values are not supported!");
        auto norm_attrs = opts.to_vector<op::normalize_attribute>();
        auto exprs      = migraphx::from_value<std::vector<sym::expr>>(vv);
        return migraphx::to_value(tune_attribute_sym(exprs, axes, norm_attrs, input_shape, m));
    }
    return value(tune_attribute(vv.to_vector<int64_t>(), axes, opts, input_shape, m));
}

/// Normalizes one entry of the `normalize_axes` map and writes it back into the operator.
static void
normalize_axes_attribute(operation& op, value& val, const value& rv, const shape& input_shape)
{
    const auto& key = rv.get_key();
    if(not val.contains(key))
        MIGRAPHX_THROW("NORMALIZE_ATTR : op " + op.name() + " attribute \"" + key +
                       "\" not exist!");
    auto message = [&] { return op.name() + ": " + key + ": "; };
    auto opts    = rv.without_key();
    auto vv      = val.at(key).without_key();
    if(vv.is_array())
    {
        std::vector<int64_t> axes;
        if(val.contains("axes"))
        {
            axes = val.at("axes").without_key().to_vector<int64_t>();
        }
        val[key] = tune_array_attribute(vv, axes, opts, input_shape, message);
    }
    else
    {
        auto num = vv.to<int64_t>();
        val[key] = tune_attribute({num}, {num}, opts, input_shape, message).front();
    }
    op.from_value(val);
    val = op.to_value();
}

/// Callers pass the shape of the operator's first input.
bool normalize_attributes(operation& op, const shape& input_shape)
{
    bool tuned = false;
    auto attrs = op.attributes();
    auto val   = op.to_value();
    if(attrs.contains("normalize_padding"))
    {
        tuned = normalize_padding_attribute(
            op, val, attrs.at("normalize_padding").to<std::string>(), input_shape);
    }
    if(not attrs.contains("normalize_axes"))
    {
        return tuned;
    }

    // A value object keeps insertion order, so an operator that lists "axes" first in its
    // normalize_axes map gets it resolved before the bounds normalized against it.
    for(const auto& rv : attrs.at("normalize_axes").without_key())
    {
        normalize_axes_attribute(op, val, rv, input_shape);
        tuned = true;
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
