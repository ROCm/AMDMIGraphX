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
#include <migraphx/split_sym/analyzer.hpp>
#include <limits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {

axis_desc padded_axis(fill_kind fill, bool coalesce_safe)
{
    return {axis_handling::pad, fill, coalesce_safe};
}

axis_desc parallel_axis() { return padded_axis(fill_kind::dont_care, true); }

axis_desc contracted_axis(fill_kind fill) { return padded_axis(fill, false); }

axis_desc masked_axis(mask_role role, fill_kind fill)
{
    return {axis_handling::mask, fill, true, role};
}

bool is_variable_axis(const shape::dynamic_dimension& d)
{
    return d.is_symbolic() and not d.is_fixed();
}

bool supports_mask(shape::type_t type, fill_kind fill)
{
    if(fill != fill_kind::neg_inf)
        return true;
    return contains({shape::half_type, shape::float_type, shape::double_type, shape::bf16_type},
                    type);
}

float fill_value(fill_kind fill)
{
    switch(fill)
    {
    case fill_kind::dont_care:
    case fill_kind::zero: return 0.0f;
    case fill_kind::lowest: return std::numeric_limits<float>::lowest();
    case fill_kind::neg_inf: return -std::numeric_limits<float>::infinity();
    case fill_kind::highest: return std::numeric_limits<float>::max();
    case fill_kind::one: return 1.0f;
    }
    MIGRAPHX_THROW("SPLIT_SYM_DIM: unsupported fill kind");
}

std::optional<std::size_t> normalize_axis(int64_t axis, std::size_t rank)
{
    if(axis < 0)
        axis += static_cast<int64_t>(rank);
    if(axis < 0 or axis >= static_cast<int64_t>(rank))
        return std::nullopt;
    return static_cast<std::size_t>(axis);
}

bool windowed_zero_pad(const std::vector<std::size_t>& padding,
                       std::size_t spatial_dimensions,
                       std::size_t axis)
{
    if(axis < 2)
        return false;
    std::size_t spatial_dimension = axis - 2;
    if(spatial_dimension >= spatial_dimensions)
        return false;
    if(padding.size() == spatial_dimensions)
        return padding[spatial_dimension] == 0;
    if(padding.size() == 2 * spatial_dimensions)
        return padding[spatial_dimension] == 0 and
               padding[spatial_dimensions + spatial_dimension] == 0;
    return false;
}

symbolic_op_info analyze_axes(instruction_ref ins, std::vector<std::size_t> shape_input_indices)
{
    return analyze_axes(
        ins,
        std::move(shape_input_indices),
        [](std::size_t) { return true; },
        [](std::size_t, std::size_t) { return parallel_axis(); });
}

static std::vector<symbolic_analyzer>& analyzers()
{
    static std::vector<symbolic_analyzer> m; // NOLINT
    return m;
}

void register_analyzer(symbolic_analyzer analyzer) { analyzers().push_back(std::move(analyzer)); }

symbolic_op_info analyze_instruction(instruction_ref ins)
{
    const symbolic_analyzer* match = nullptr;
    for(const auto& analyzer : analyzers())
    {
        if(not analyzer.matches(ins->get_operator()))
            continue;
        assert(match == nullptr);
        if(match == nullptr)
            match = &analyzer;
    }
    if(match == nullptr)
        return symbolic_op_info{ins};
    return match->analyze(ins);
}

} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
