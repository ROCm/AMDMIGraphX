#include <migraphx/reshape_dims.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator>
    static auto compute_end_dim(Iterator start, Iterator last, std::size_t dim)
    {
        std::size_t x = 1;
        auto it       = std::find_if(start, last, [&](auto i) {
            x *= i;
            return x >= dim;
        });
        if(x != dim)
            return start;
        return it;
    }

    template <class OptionalPair>
    static OptionalPair try_merge_pairs(OptionalPair p2, OptionalPair p1)
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
        // Transposed
        if(stride2 > stride1)
            return nullopt;
        // Broadcasted check to avoid division by zero
        if(stride2 == 0)
        {
            if(stride1 == 0)
                return {{elements, 0}};
            return nullopt;
        }
        if(stride1 % stride2 != 0)
            return nullopt;
        auto space = (stride1 * dim1 + stride2 * dim2 - stride1) / stride2;
        // Nonpacked
        if(space != elements)
            return nullopt;
        return {{elements, stride2}};
    }

    template <class DimIterator, class StrideIterator>
    static optional<std::size_t> merge_strides(DimIterator dim_start,
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

    optional<shape> reshape_dims(const shape& input,
                            const std::vector<std::size_t>& rdims, reshape_dims_options options)
    {
        if(input.standard())
            return shape{input.type(), rdims};

        const auto& idims    = input.lens();
        const auto& istrides = input.strides();

        std::vector<std::size_t> rstrides;
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
            // squeeze
            else if(rdim > idim)
            {
                auto start = idims.begin() + i;
                auto it    = compute_end_dim(start, idims.end(), rdim);
                if(it == start)
                    return nullopt;
                auto n = it - start;
                assert((i + n) <= istrides.size());
                if(options.lazy and not can_strides_merge(
                       start, it + 1, istrides.begin() + i, istrides.begin() + i + n + 1))
                    return nullopt;
                i += n;
                rstrides.push_back(istrides[i]);
            }
            // unsqueeze
            else // if(rdim < idim)
            {
                auto start = rdims.begin() + i;
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
            i++;
            r++;
        }

        // Handle trailing 1s
        if(rstrides.size() < rdims.size() and not rstrides.empty())
        {
            auto stride = rstrides.back();
            for(auto d : range(rdims.begin() + rstrides.size(), rdims.end()))
            {
                if(d != 1)
                    return nullopt;
                rstrides.push_back(stride);
            }
        }

        if(rdims.size() != rstrides.size())
            return nullopt;

        auto result = shape{input.type(), rdims, rstrides};
        if(options.lazy or result.packed())
            return result;
        // TODO: Add as_packed to shape class
        return result.with_lens(result.type(), result.lens());
    }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx


