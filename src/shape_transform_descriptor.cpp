#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/common_dims.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Range>
static auto elements(const Range& r)
{
    return std::accumulate(r.begin(), r.end(), std::size_t{1}, std::multiplies<>{});
}

template <class Iterator, class Projection>
static auto compute_end_dim(Iterator start, Iterator last, std::size_t dim, Projection proj)
{
    std::size_t x = 1;
    auto it       = std::find_if(start, last, [&](auto d) {
        x *= proj(d);
        return x >= dim;
    });
    if(x != dim)
        return start;
    return it;
}

shape_transform_descriptor::shape_transform_descriptor(const std::vector<std::size_t>& dims)
{
    transform(dims,
              range(dims.size()),
              std::back_inserter(dimensions),
              [](std::size_t d, std::size_t a) -> dimension {
                  return {{dimension::sub{d, {a}}}};
              });
}

std::vector<shape_transform_descriptor::dimension::sub>
shape_transform_descriptor::get_all_subdimensions() const
{
    std::vector<dimension::sub> result;
    for(const auto& dim : dimensions)
    {
        result.insert(result.end(), dim.subdimensions.begin(), dim.subdimensions.end());
    }
    return result;
}

bool shape_transform_descriptor::apply(const std::vector<operation>& ops)
{
    for(const auto& op : ops)
    {
        auto v = op.to_value();
        if(op.name() == "reshape")
        {
            if(not apply_reshape(v["dims"].to_vector<std::size_t>()))
                return false;
        }
        else if(op.name() == "transpose")
        {
            if(not apply_transpose(v["permutation"].to_vector<std::int64_t>()))
                return false;
        }
        else if(op.name() == "multibroadcast")
        {
            if(not apply_broadcast(v["out_lens"].to_vector<std::size_t>()))
                return false;
        }
        else
        {
            return false;
        }
    }
    return true;
}
bool shape_transform_descriptor::apply_reshape(const std::vector<std::size_t>& rdims)
{
    assert(migraphx::elements(rdims) == this->elements());
    std::vector<dimension> new_dims;
    auto subs     = get_all_subdimensions();
    std::size_t i = 0;
    std::size_t r = 0;
    while(i < subs.size() and r < rdims.size())
    {
        const auto& sub = subs[i];
        auto idim       = sub.len;
        auto rdim       = rdims[r];
        if(idim == rdim)
        {
            new_dims.push_back({{sub}});
        }
        // squeeze
        else if(rdim > idim)
        {
            auto start = subs.begin() + i;
            auto it = compute_end_dim(start, subs.end(), rdim, std::mem_fn(&dimension::sub::len));
            if(it == start)
                return false;
            auto n = it - start;
            i += n;
            new_dims.push_back({{start, it + 1}});
        }
        // unsqueeze
        else // if(rdim < idim)
        {
            auto start = rdims.begin() + i;
            auto it    = compute_end_dim(start, rdims.end(), idim, id{});
            if(it == start)
                return false;
            auto n = it - start;
            r += n;
            transform(range(n + 1), std::back_inserter(new_dims), [&](auto j) -> dimension {
                auto new_sub = sub;
                if(not new_sub.axis.empty())
                    new_sub.axis.push_back(j);
                new_sub.len = start[j];
                return {{new_sub}};
            });
        }
        r++;
        i++;
    }

    // Handle trailing 1s
    if(new_dims.size() < rdims.size() and not new_dims.empty())
    {
        for(auto d : range(rdims.begin() + new_dims.size(), rdims.end()))
        {
            if(d != 1)
                return false;
            new_dims.push_back({{dimension::sub{1}}});
        }
    }

    if(rdims.size() != new_dims.size())
        return false;
    dimensions = new_dims;
    return true;
}
bool shape_transform_descriptor::apply_transpose(const std::vector<std::int64_t>& permutation)
{
    if(permutation.size() != dimensions.size())
        return false;
    dimensions = reorder_dims(dimensions, permutation);
    return true;
}

bool shape_transform_descriptor::apply_broadcast(const std::vector<std::size_t>& out_lens,
                                                 optional<std::size_t> axis)
{
    auto offset = out_lens.size() - dimensions.size();
    std::vector<dimension> new_dims;
    std::transform(out_lens.begin(),
                   out_lens.begin() + offset,
                   std::back_inserter(new_dims),
                   [&](auto len) -> dimension {
                       return {{dimension::sub{len, {}}}};
                   });
    std::transform(out_lens.begin() + offset,
                   out_lens.end(),
                   dimensions.begin(),
                   std::back_inserter(new_dims),
                   [&](auto len, const dimension& dim) -> dimension {
                       if(len == dim.len())
                           return dim;
                       if(dim.len() != 1)
                           MIGRAPHX_THROW("Wrong out_lens for broadcast");
                       return {{dimension::sub{len, {}}}};
                   });
    dimensions = new_dims;
    return true;
}

std::size_t shape_transform_descriptor::dimension::len() const
{
    return transform_accumulate(subdimensions.begin(),
                                subdimensions.end(),
                                std::size_t{1},
                                std::multiplies<>{},
                                [](const auto& s) { return s.len; });
}

std::size_t shape_transform_descriptor::elements() const
{
    return transform_accumulate(dimensions.begin(),
                                dimensions.end(),
                                std::size_t{1},
                                std::multiplies<>{},
                                [](const auto& s) { return s.len(); });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
