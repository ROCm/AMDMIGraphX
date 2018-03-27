#ifndef GUARD_RTGLIB_LITERAL_HPP
#define GUARD_RTGLIB_LITERAL_HPP

#include <rtg/shape.hpp>

namespace rtg {

struct literal 
{
    std::vector<char> buffer;
    shape shape_;
};

}

#endif
