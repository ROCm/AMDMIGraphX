#ifndef MIGRAPH_GUARD_RTGLIB_CHECK_SHAPES_HPP
#define MIGRAPH_GUARD_RTGLIB_CHECK_SHAPES_HPP

#include <migraph/shape.hpp>
#include <algorithm>

namespace migraph {

struct check_shapes
{
    const std::vector<shape>* shapes;
    const std::string name;

    check_shapes(const std::vector<shape>& s) : shapes(&s) {}

    template <class Op>
    check_shapes(const std::vector<shape>& s, const Op& op) : shapes(&s), name(op.name())
    {
    }

    std::string prefix() const
    {
        if(name.empty())
            return "";
        else
            return name + ": ";
    }

    const check_shapes& has(std::size_t n) const
    {
        assert(shapes != nullptr);
        if(shapes->size() != n)
            MIGRAPH_THROW(prefix() + "Wrong number of arguments: expected " + std::to_string(n) +
                          " but given " + std::to_string(shapes->size()));
        return *this;
    }

    const check_shapes& only_dims(std::size_t n) const
    {
        assert(shapes != nullptr);
        if(!shapes->empty())
        {
            if(shapes->front().lens().size() != n)
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
        assert(shapes != nullptr);
        if(shapes->empty())
            return true;
        auto&& key = f(shapes->front());
        return this->all_of([&](const shape& s) { return f(s) == key; });
    }

    template <class Predicate>
    bool all_of(Predicate p) const
    {
        assert(shapes != nullptr);
        return std::all_of(shapes->begin(), shapes->end(), p);
    }
};

} // namespace migraph

#endif
