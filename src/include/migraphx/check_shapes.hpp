#ifndef MIGRAPH_GUARD_RTGLIB_CHECK_SHAPES_HPP
#define MIGRAPH_GUARD_RTGLIB_CHECK_SHAPES_HPP

#include <migraphx/shape.hpp>
#include <migraphx/config.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

struct check_shapes
{
    const shape* begin;
    const shape* end;
    const std::string name;

    check_shapes(const shape* b, const shape* e, const std::string& n) : begin(b), end(e), name(n)
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

    const check_shapes& has(std::size_t n) const
    {
        if(size() != n)
            MIGRAPH_THROW(prefix() + "Wrong number of arguments: expected " + std::to_string(n) +
                          " but given " + std::to_string(size()));
        return *this;
    }

    const check_shapes& only_dims(std::size_t n) const
    {
        assert(begin != nullptr);
        assert(end != nullptr);
        if(begin != end)
        {
            if(begin->lens().size() != n)
                MIGRAPH_THROW(prefix() + "Only " + std::to_string(n) + "d supported");
        }
        return *this;
    }

    const check_shapes& same_shape() const
    {
        if(!this->same([](const shape& s) { return s; }))
            MIGRAPH_THROW(prefix() + "Shapes do not match");
        return *this;
    }

    const check_shapes& same_type() const
    {
        if(!this->same([](const shape& s) { return s.type(); }))
            MIGRAPH_THROW(prefix() + "Types do not match");
        return *this;
    }

    const check_shapes& same_dims() const
    {
        if(!this->same([](const shape& s) { return s.lens(); }))
            MIGRAPH_THROW(prefix() + "Dimensions do not match");
        return *this;
    }

    const check_shapes& same_ndims() const
    {
        if(!this->same([](const shape& s) { return s.lens().size(); }))
            MIGRAPH_THROW(prefix() + "Number of dimensions do not match");
        return *this;
    }

    const check_shapes& standard() const
    {
        if(!this->all_of([](const shape& s) { return s.standard(); }))
            MIGRAPH_THROW(prefix() + "Shapes are not in standard layout");
        return *this;
    }

    const check_shapes& packed() const
    {
        if(!this->all_of([](const shape& s) { return s.packed(); }))
            MIGRAPH_THROW(prefix() + "Shapes are not packed");
        return *this;
    }

    const check_shapes& not_transposed() const
    {
        if(!this->all_of([](const shape& s) { return not s.transposed(); }))
            MIGRAPH_THROW(prefix() + "Shapes are transposed");
        return *this;
    }

    const check_shapes& not_broadcasted() const
    {
        if(!this->all_of([](const shape& s) { return not s.broadcasted(); }))
            MIGRAPH_THROW(prefix() + "Shapes are broadcasted");
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
            MIGRAPH_THROW(prefix() + "Accessing shape out of bounds");
        assert(begin != nullptr);
        assert(end != nullptr);
        if(i < 0)
            return end - i;
        return begin + i;
    }

    check_shapes slice(long start) { return {get(start), end, name}; }

    check_shapes slice(long start, long last) { return {get(start), get(last), name}; }
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
