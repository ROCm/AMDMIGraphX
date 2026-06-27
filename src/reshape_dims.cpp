/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#include <migraphx/reshape_dims.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator>
static Iterator compute_end_dim(Iterator start, Iterator last, const sym::expr& dim)
{
    auto x             = sym::lit(std::int64_t{1});
    bool indeterminate = false;
    auto it            = std::find_if(start, last, [&](const auto& i) {
        x *= i;
        auto x_lt = sym::strict_less(x, dim);
        if(not x_lt.has_value())
        {
            indeterminate = true;
            return true;
        }
        return not *x_lt;
    });
    if(indeterminate or not sym::same_symbol(x, dim))
        return start;
    return it;
}

static optional<std::pair<sym::expr, sym::expr>>
try_merge_pairs(optional<std::pair<sym::expr, sym::expr>> p2,
                optional<std::pair<sym::expr, sym::expr>> p1)
{
    if(not p1.has_value())
        return nullopt;
    if(not p2.has_value())
        return nullopt;
    auto dim1     = p1->first;
    auto dim2     = p2->first;
    auto stride1  = p1->second;
    auto stride2  = p2->second;
    auto elements = dim1 * dim2;
    auto zero     = sym::lit(std::int64_t{0});
    // Transposed
    auto order = sym::strict_less(stride1, stride2);
    if(not order.has_value() or *order)
        return nullopt;
    // Broadcasted check to avoid division by zero
    if(sym::same_symbol(stride2, zero))
    {
        if(sym::same_symbol(stride1, zero))
            return {{elements, zero}};
        return nullopt;
    }
    if(not sym::same_symbol(stride1 % stride2, zero))
        return nullopt;
    auto space = (stride1 * dim1 + stride2 * dim2 - stride1) / stride2;
    // Nonpacked
    if(not sym::same_symbol(space, elements))
        return nullopt;
    return {{elements, stride2}};
}

template <class DimIterator, class StrideIterator>
static optional<sym::expr> merge_strides(DimIterator dim_start,
                                         DimIterator dim_last,
                                         StrideIterator stride_start,
                                         StrideIterator stride_last)
{
    if(dim_start == dim_last)
        return nullopt;
    (void)stride_start; // Is only used in the assert
    assert(std::distance(dim_start, dim_last) == std::distance(stride_start, stride_last));
    auto make_pair_optional = [&](auto dim, auto stride) {
        return std::make_optional(std::make_pair(dim, stride));
    };
    auto dim_stride_pair =
        std::inner_product(std::make_reverse_iterator(dim_last - 1),
                           std::make_reverse_iterator(dim_start),
                           std::make_reverse_iterator(stride_last - 1),
                           make_pair_optional(*std::prev(dim_last), *std::prev(stride_last)),
                           MIGRAPHX_LIFT(try_merge_pairs),
                           make_pair_optional);
    if(not dim_stride_pair.has_value())
        return nullopt;
    return dim_stride_pair->second;
}

template <class DimIterator, class StrideIterator>
static auto can_strides_merge(DimIterator dim_start,
                              DimIterator dim_last,
                              StrideIterator stride_start,
                              StrideIterator stride_last)
{
    return merge_strides(dim_start, dim_last, stride_start, stride_last).has_value();
}

std::vector<shape::dynamic_dimension> resolve_reshape_dims(const shape& sym_in,
                                                           const std::vector<dim_like>& dims)
{
    const auto& input_dds = sym_in.dyn_dims();
    std::vector<shape::dynamic_dimension> output_dyn_dims(dims.size());
    shape::dynamic_dimension known_elements{sym::lit(1)};
    std::size_t neg_dim_num = dims.size();
    for(std::size_t i = 0; i < dims.size(); ++i)
    {
        const auto& d = dims[i];
        // Defer -1; it needs the product of every other axis.
        if(d == dim_like{-1})
        {
            neg_dim_num = i;
            continue;
        }
        if(d == dim_like{0})
            output_dyn_dims[i] = input_dds.at(i);
        else if(is_symbolic(d))
            output_dyn_dims[i] = std::get<shape::dynamic_dimension>(d);
        else
            output_dyn_dims[i] = shape::dynamic_dimension{sym::lit(std::get<int64_t>(d))};
        known_elements = known_elements * output_dyn_dims[i];
    }
    if(neg_dim_num < dims.size())
    {
        auto total_elements          = std::accumulate(input_dds.begin(),
                                              input_dds.end(),
                                              shape::dynamic_dimension{sym::lit(1)},
                                              std::multiplies<>{});
        output_dyn_dims[neg_dim_num] = total_elements / known_elements;
    }
    return output_dyn_dims;
}

void validate_reshape_dims(const std::string& name, const std::vector<dim_like>& dims)
{
    if(std::any_of(dims.begin(), dims.end(), [](const dim_like& d) {
           return std::holds_alternative<shape::dynamic_dimension>(d) and not is_symbolic(d);
       }))
        MIGRAPHX_THROW(name + ": dim entries must be int64 or symbolic");

    auto n_neg_dims = std::count(dims.begin(), dims.end(), dim_like{-1});
    if(n_neg_dims > 1)
        MIGRAPHX_THROW(name + ": Dimensions for " + name + " can only have one -1 dim but given {" +
                       to_string_range(dims) + "} with " + to_string(n_neg_dims) + " -1 dims");
}

optional<shape>
reshape_dims(const shape& input, const std::vector<sym::expr>& rdims, reshape_dims_options options)
{
    const std::vector<shape::dynamic_dimension> rdds(rdims.begin(), rdims.end());

    if(input.standard())
        return shape{input.type(), rdds};

    // Broadcasts have ambiguous permutations (multiple axes share stride 0), so
    // for non-lazy reshape fall back to a standard layout. Sliced (non-packed)
    // inputs still propagate the permutation via the algorithm + with_lens below.
    if(not options.lazy and input.broadcasted())
        return shape{input.type(), rdds};

    std::vector<sym::expr> idims(input.dyn_dims().size());
    std::transform(input.dyn_dims().begin(),
                   input.dyn_dims().end(),
                   idims.begin(),
                   [](const auto& dd) { return dd.sym_expr; });
    const auto& istrides = input.dyn_strides();

    std::vector<sym::expr> rstrides;
    std::size_t i = 0;
    std::size_t r = 0;
    while(i < idims.size() and r < rdims.size())
    {
        auto idim = idims[i];
        auto rdim = rdims[r];
        if(rdim == idim)
        {
            rstrides.push_back(istrides[i]);
        }
        else
        {
            // == handled above; an unprovable ordering bails.
            auto rdim_gt = sym::strict_less(idim, rdim);
            auto rdim_lt = sym::strict_less(rdim, idim);
            if(not rdim_gt.has_value() or not rdim_lt.has_value())
                return nullopt;
            // squeeze
            if(*rdim_gt)
            {
                auto start = idims.begin() + i;
                auto it    = compute_end_dim(start, idims.end(), rdim);
                if(it == start)
                    return nullopt;
                auto n = it - start;
                assert((i + n) <= istrides.size());
                if(options.lazy and
                   not can_strides_merge(
                       start, it + 1, istrides.begin() + i, istrides.begin() + i + n + 1))
                    return nullopt;
                i += n;
                rstrides.push_back(istrides[i]);
            }
            // unsqueeze
            else if(*rdim_lt)
            {
                auto start = rdims.begin() + r;
                auto it    = compute_end_dim(start, rdims.end(), idim);
                if(it == start)
                    return nullopt;
                auto n = it - start;
                assert((r + n) <= rdims.size());
                auto stride = istrides[i] * idim;
                std::for_each(start, it + 1, [&](auto dim) {
                    stride /= dim;
                    rstrides.push_back(stride);
                });
                r += n;
            }
            else
            {
                return nullopt;
            }
        }
        i++;
        r++;
    }

    // Handle trailing 1s
    if(rstrides.size() < rdims.size() and not rstrides.empty())
    {
        auto stride = rstrides.back();
        for(auto d : range(rdims.begin() + rstrides.size(), rdims.end()))
        {
            if(d != sym::lit(std::int64_t{1}))
                return nullopt;
            rstrides.push_back(stride);
        }
    }

    if(rdims.size() != rstrides.size())
        return nullopt;

    auto result = shape{input.type(), rdds, rstrides};
    if(options.lazy or result.packed())
        return result;
    // TODO: Add as_packed to shape class
    return result.with_lens(result.type(), result.dyn_dims());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
