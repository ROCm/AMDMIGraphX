#ifndef GUARD_RTGLIB_ARGUMENT_HPP
#define GUARD_RTGLIB_ARGUMENT_HPP

#include <rtg/shape.hpp>
#include <functional>

namespace rtg {

struct argument
{
    std::function<char*()> data;
    shape s;

    template<class Visitor>
    void visit(Visitor v) const
    {
        s.visit_type([&](auto as) {
            v(as.from(data()));
        });
    }
};

}

#endif
