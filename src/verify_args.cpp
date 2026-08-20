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

#include <migraphx/verify_args.hpp>
#include <migraphx/env.hpp>
#include <migraphx/fp8_types.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/ranges.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_VERIFY_DUMP_DIFF);

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Spacing between representable values at 1.0. Returns zero for the integral types, which are
/// compared exactly, and for tuples, which have no epsilon at all.
static double type_epsilon(shape::type_t type)
{
    // fp4x2 cannot be visited, and e2m1 carries a single mantissa bit.
    if(type == shape::fp4x2_type)
        return 0.5;
    double result = 0;
    if(type != shape::tuple_type and shape::is_computable(type))
        shape::visit(type, [&](auto as) { result = as.epsilon(); });
    return result;
}

/// Rounding steps of the output type the target may differ from the reference by. Multiplying
/// epsilon gives a bound that is this many steps at every magnitude, since the reference magnitude
/// cancels against the widening spacing between representable values.
static constexpr double rtol_ulps = 4;
/// Where the reference is near zero the relative term collapses and only atol applies. Held to a
/// fraction of a rounding step so that it governs those elements and nothing else.
static constexpr double atol_ulps = 0.5;

verify::tolerance tolerance_for_type(shape::type_t type)
{
    // float and wider keep the historical fixed bounds. They are far looser than a few rounding
    // steps, but a great many tests are calibrated to them, and a gpu library kernel legitimately
    // differs from the naive ref implementation by more than 4 fp32 steps.
    auto eps = type_epsilon(type);
    if(type == shape::fp4x2_type)
        return {8e-1, atol_ulps * eps, rtol_ulps * eps};
    if(contains(fp8_types{}.get(), type))
        return {2e-1, atol_ulps * eps, rtol_ulps * eps};
    if(type == shape::half_type or type == shape::bf16_type)
        return {8e-2, atol_ulps * eps, rtol_ulps * eps};
    return {};
}

verify::tolerance default_tolerance_for(shape::type_t type, std::size_t rms_multiplier)
{
    auto tols = tolerance_for_type(type);
    // fp4x2 and tuple keep the rms bound from the precision class: the former is compared as raw
    // bytes and the latter has no epsilon at all. Integral types scale from an epsilon of zero,
    // which is what holds them to an exact match.
    if(type != shape::tuple_type and shape::is_computable(type))
        tols.rms_tol = type_epsilon(type) * rms_multiplier;
    return tols;
}

/// Factor atol alone would have to grow by for every element to pass, holding rtol fixed. This is
/// the knob for elements whose reference is at or near zero, where the relative term contributes
/// nothing.
template <class R1, class R2>
static double required_atol_scale(const R1& ref, const R2& target, verify::tolerance tols)
{
    return verify::range_product(ref, target, 0.0, verify::max, [&](auto r, auto t) {
        auto need = verify::abs_diff(double(t), double(r)) - tols.rtol * std::fabs(double(r));
        if(need <= 0)
            return 0.0;
        return tols.atol > 0 ? need / tols.atol : std::numeric_limits<double>::infinity();
    });
}

/// Factor rtol alone would have to grow by, holding atol fixed. Infinite when a failing element
/// has a reference of zero: no relative bound can cover that, so the override has to go to atol.
template <class R1, class R2>
static double required_rtol_scale(const R1& ref, const R2& target, verify::tolerance tols)
{
    return verify::range_product(ref, target, 0.0, verify::max, [&](auto r, auto t) {
        auto need = verify::abs_diff(double(t), double(r)) - tols.atol;
        if(need <= 0)
            return 0.0;
        auto scaled = tols.rtol * std::fabs(double(r));
        return scaled > 0 ? need / scaled : std::numeric_limits<double>::infinity();
    });
}

/// How widespread the disagreement is. A handful of elements over the bound is rounding that
/// the tolerance is too tight for; most of the tensor over the bound is a different result.
template <class R1, class R2>
static std::size_t count_outside_tolerance(const R1& ref, const R2& target, verify::tolerance tols)
{
    return verify::range_product(
        ref, target, std::size_t{0}, verify::sum, [&](auto r, auto t) -> std::size_t {
            auto bound = tols.atol + tols.rtol * std::fabs(double(r));
            return verify::abs_diff(double(t), double(r)) < bound ? 0 : 1;
        });
}

bool verify_args(const std::string& name,
                 const argument& target_arg,
                 const verify::expected<argument>& ref_arg,
                 verify::tolerance tols)
{
    bool passed    = true;
    argument t_arg = target_arg;
    argument r_arg = ref_arg.data();
    if(not t_arg.get_shape().computable())
    {
        shape o_t_shape = t_arg.get_shape();
        shape o_r_shape = r_arg.get_shape();
        assert(o_t_shape.type() == o_r_shape.type());
        t_arg = t_arg.reshape(shape{shape::uint8_type, o_t_shape.lens(), o_t_shape.strides()});
        r_arg = r_arg.reshape(shape{shape::uint8_type, o_r_shape.lens(), o_r_shape.strides()});
    }
    visit_all(r_arg, t_arg)([&](auto ref, auto target) {
        double rms_error;
        passed =
            verify::verify_range_with_tolerance(target, verify::expected{ref}, tols, &rms_error);
        if(not passed)
        {
            // TODO: Check for nans
            log::error() << "FAILED: " << name;
            log::error() << "RMS Error: " << rms_error;
            if(ref.size() < 32 or enabled(MIGRAPHX_VERIFY_DUMP_DIFF{}))
                log::error() << "ref:" << ref;
            if(target.size() < 32 or enabled(MIGRAPHX_VERIFY_DUMP_DIFF{}))
                log::error() << "target:" << target;
            if(verify::range_zero(ref))
                log::error() << "Ref data is all zeros";
            if(verify::range_zero(target))
                log::error() << "Target data is all zeros";

            auto mxdiff = verify::max_diff(ref, target);
            log::error() << "Max diff: " << mxdiff;
            log::error() << "Tolerances: atol=" << tols.atol << " rtol=" << tols.rtol
                         << " rms_tol=" << tols.rms_tol;
            log::error() << "Required atol scale: " << required_atol_scale(ref, target, tols)
                         << " (rtol alone: " << required_rtol_scale(ref, target, tols) << ")";
            log::error() << "Elements outside tolerance: "
                         << count_outside_tolerance(ref, target, tols) << " / " << ref.size();

            auto idx = verify::mismatch_idx(ref, target, float_equal);
            if(idx < verify::range_distance(ref))
            {
                log::error() << "Mismatch at " << idx << ": " << ref[idx] << " != " << target[idx];
            }

            auto ref_nan_idx = find_idx(ref, verify::not_finite);
            if(ref_nan_idx >= 0)
                log::error() << "Non finite number found in ref at " << ref_nan_idx << ": "
                             << ref[ref_nan_idx];

            auto target_nan_idx = find_idx(target, verify::not_finite);
            if(target_nan_idx >= 0)
                log::error() << "Non finite number found in target at " << target_nan_idx << ": "
                             << target[target_nan_idx];
        }
        else
        {
            if(verify::range_zero(ref))
                log::warn() << "Ref data is all zeros";
            if(verify::range_zero(target))
                log::warn() << "Target data is all zeros";

            auto ref_nan_idx = find_idx(ref, verify::not_finite);
            if(ref_nan_idx >= 0)
                log::warn() << "Non finite number found in ref at " << ref_nan_idx << ": "
                            << ref[ref_nan_idx];

            auto target_nan_idx = find_idx(target, verify::not_finite);
            if(target_nan_idx >= 0)
                log::warn() << "Non finite number found in target at " << target_nan_idx << ": "
                            << target[target_nan_idx];
        }
    });
    return passed;
}

bool verify_args_with_tolerance(const std::string& name,
                                const argument& target_arg,
                                const verify::expected<argument>& ref_arg,
                                optional<verify::tolerance> tols)
{
    if(not tols.has_value())
    {
        // Non-computable types are compared as raw bytes, so their bound comes from the byte type.
        auto type =
            target_arg.get_shape().computable() ? target_arg.get_shape().type() : shape::uint8_type;
        tols = default_tolerance_for(type);
    }
    return verify_args(name, target_arg, ref_arg, *tols);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
