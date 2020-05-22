#ifndef MIGRAPHX_GUARD_RTGLIB_CHECK_SHAPES_HPP
#define MIGRAPHX_GUARD_RTGLIB_CHECK_SHAPES_HPP

#include <migraphx/shape.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/config.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct check_shapes
{
    const shape* begin;
    const shape* end;
    const std::string name;

    check_shapes(const shape* b, const shape* e, const std::string& n) : begin(b), end(e), name(n)
    {
    }

    template <class Op>
    check_shapes(const shape* b, const shape* e, const Op& op) : begin(b), end(e), name(op.name())
    {
    }

    check_shapes(const std::vector<shape>& s) : begin(s.data()), end(s.data() + s.size()) {}

    template <class Op>
    check_shapes(const std::vector<shape>& s, const Op& op)
        : begin(s.data()), end(s.data() + s.size()), name(op.name())
    {
    }

    std::string prefix() const
    {
        if(name.empty())
            return "";
        else
            return name + ": ";
    }

    std::size_t size() const
    {
        if(begin == end)
            return 0;
        assert(begin != nullptr);
        assert(end != nullptr);
        return end - begin;
    }

    template <class... Ts>
    const check_shapes& has(Ts... ns) const
    {
        if(migraphx::none_of({ns...}, [&](auto i) { return this->size() == i; }))
            MIGRAPHX_THROW(prefix() + "Wrong number of arguments: expected " +
                           to_string_range({ns...}) + " but given " + std::to_string(size()));
        return *this;
    }

    const check_shapes& only_dims(std::size_t n) const
    {
        assert(begin != nullptr);
        assert(end != nullptr);
        if(begin != end)
        {
            if(begin->lens().size() != n)
                MIGRAPHX_THROW(prefix() + "Only " + std::to_string(n) + "d supported");
        }
        return *this;
    }

    const check_shapes& min_ndims(std::size_t n) const
    {
        assert(begin != nullptr);
        assert(end != nullptr);
        if(begin != end)
        {
            if(begin->lens().size() < n)
                MIGRAPHX_THROW(prefix() + "Shape must have at least " + std::to_string(n) + " dimensions");
        }
        return *this;
    }

    const check_shapes& same_shape() const
    {
        if(!this->same([](const shape& s) { return s; }))
            MIGRAPHX_THROW(prefix() + "Shapes do not match");
        return *this;
    }

    const check_shapes& same_type() const
    {
        if(!this->same([](const shape& s) { return s.type(); }))
            MIGRAPHX_THROW(prefix() + "Types do not match");
        return *this;
    }

    const check_shapes& same_dims() const
    {
        if(!this->same([](const shape& s) { return s.lens(); }))
            MIGRAPHX_THROW(prefix() + "Dimensions do not match");
        return *this;
    }

    const check_shapes& same_ndims() const
    {
        if(!this->same([](const shape& s) { return s.lens().size(); }))
            MIGRAPHX_THROW(prefix() + "Number of dimensions do not match");
        return *this;
    }

    const check_shapes& standard() const
    {
        if(!this->all_of([](const shape& s) { return s.standard(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are not in standard layout");
        return *this;
    }

    const check_shapes& standard_or_scalar() const
    {
        if(!this->all_of([](const shape& s) { return s.standard() or s.scalar(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are not a scalar or in standard layout");
        return *this;
    }

    const check_shapes& packed() const
    {
        if(!this->all_of([](const shape& s) { return s.packed(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are not packed");
        return *this;
    }

    const check_shapes& not_transposed() const
    {
        if(!this->all_of([](const shape& s) { return not s.transposed(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are transposed");
        return *this;
    }

    const check_shapes& not_broadcasted() const
    {
        if(!this->all_of([](const shape& s) { return not s.broadcasted(); }))
            MIGRAPHX_THROW(prefix() + "Shapes are broadcasted");
        return *this;
    }

    const check_shapes& elements(std::size_t n) const
    {
        if(!this->all_of([&](const shape& s) { return s.elements() == n; }))
            MIGRAPHX_THROW(prefix() + "Wrong number of elements");
        return *this;
    }

    template <class F>
    bool same(F f) const
    {
        if(begin == end)
            return true;
        assert(begin != nullptr);
        assert(end != nullptr);
        auto&& key = f(*begin);
        return this->all_of([&](const shape& s) { return f(s) == key; });
    }

    template <class Predicate>
    bool all_of(Predicate p) const
    {
        if(begin == end)
            return true;
        assert(begin != nullptr);
        assert(end != nullptr);
        return std::all_of(begin, end, p);
    }

    const shape* get(long i)
    {
        if(i >= size())
            MIGRAPHX_THROW(prefix() + "Accessing shape out of bounds");
        assert(begin != nullptr);
        assert(end != nullptr);
        if(i < 0)
            return end - i;
        return begin + i;
    }

    check_shapes slice(long start) { return {get(start), end, name}; }

    check_shapes slice(long start, long last) { return {get(start), get(last), name}; }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
