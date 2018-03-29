#ifndef GUARD_RTGLIB_ARGUMENT_HPP
#define GUARD_RTGLIB_ARGUMENT_HPP

#include <rtg/shape.hpp>
#include <functional>

namespace rtg {

struct argument
{
    argument()
    {}

    argument(shape s, std::function<char*()> d)
    : data(d), shape_(s)
    {}

    std::function<char*()> data;

    const shape& get_shape() const
    {
        return this->shape_;
    }

    template<class Visitor>
    void visit_at(Visitor v, std::size_t n=0) const
    {
        shape_.visit_type([&](auto as) {
            v(*(as.from(this->data())+shape_.index(n)));
        });
    }

    template<class Visitor>
    void visit(Visitor v) const
    {
        shape_.visit_type([&](auto as) {
            v(make_view(this->shape_, as.from(this->data())));
        });
    }
private:
    shape shape_;
};

}

#endif
