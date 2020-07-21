#include <migraphx/reduce_dims.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool reduce_dim(std::vector<shape>& shapes, std::size_t n)
{
    std::vector<std::size_t> new_lens;
    for(const auto& s : shapes)
    {
        assert(n < s.lens().size());
        if((n + 1) >= s.lens().size())
            return false;
        auto astride = s.strides()[n];
        auto alen    = s.lens()[n];
        auto bstride = s.strides()[n + 1];
        auto blen    = s.lens()[n + 1];

        if(astride == bstride * blen)
        {
            new_lens.push_back(alen * blen);
        }
    }
    if(new_lens.size() != shapes.size())
        return false;
    std::size_t i = 0;
    for(auto& s : shapes)
    {
        auto lens    = s.lens();
        auto strides = s.strides();
        lens.erase(lens.begin() + n);
        strides.erase(strides.begin() + n);
        lens[n] = new_lens[i];
        s       = shape{s.type(), lens, strides};
        i++;
    }
    return true;
}

std::size_t reduce_dim_all(std::vector<shape>& shapes, std::size_t n)
{
    while(reduce_dim(shapes, n))
    {
        assert(n < shapes.size());
    }

    return n + 1;
}
void reduce_dim_all(std::vector<shape>& shapes)
{
    std::size_t n = 0;
    while(n < shapes.front().lens().size() - 1)
        n = reduce_dim_all(shapes, n);
}

std::vector<std::size_t> base_lens(const std::vector<shape>& shapes)
{
    return std::accumulate(
        shapes.begin() + 1, shapes.end(), shapes.front().lens(), [](auto&& lens, auto&& s) {
            std::vector<std::size_t> result;
            const auto* x = &s.lens();
            const auto* y = &lens;
            if(x->size() > y->size())
                std::swap(x, y);
            std::transform(
                x->begin(), x->end(), y->begin(), std::back_inserter(result), [&](auto a, auto b) {
                    return std::max(a, b);
                });
            return result;
        });
}

shape mask_shape(const shape& s, const std::vector<std::size_t>& lens)
{
    assert(s.lens().size() == lens.size());
    std::vector<std::size_t> rstrides(lens.size());
    std::size_t stride = 1;
    for(std::size_t i = lens.size() - 1; i < lens.size(); i--)
    {
        if(lens[i] == s.lens()[i])
        {
            rstrides[i] = stride;
            stride *= lens[i];
        }
        else if(lens[i] != 1 and s.lens()[i] != 1)
        {
            return shape{};
        }
    }
    return shape{s.type(), lens, rstrides};
}

std::vector<shape> reduce_dims(const std::vector<shape>& shapes)
{
    if(shapes.empty())
        return {};
    auto result = shapes;
    auto base   = base_lens(shapes);
    for(auto&& s : shapes)
    {
        if(s.lens().size() != base.size())
            return shapes;
        if(s.lens() == base)
            continue;
        auto mshape = mask_shape(s, base);
        if(mshape.lens().size() != base.size())
            return shapes;
        result.push_back(mshape);
    }
    reduce_dim_all(result);
    result.erase(result.begin() + shapes.size(), result.end());
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
