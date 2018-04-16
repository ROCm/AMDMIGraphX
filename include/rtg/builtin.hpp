#ifndef RTG_GUARD_BUILTIN_HPP
#define RTG_GUARD_BUILTIN_HPP

#include <rtg/operand.hpp>

namespace rtg {

namespace builtin {

struct literal
{
    std::string name() const
    {
        return "@literal";
    }
    shape compute_shape(std::vector<shape> input) const
    {
        throw "builtin"; 
    }
    argument compute(std::vector<argument> input) const
    {
        throw "builtin";
    }
};

struct param
{
    std::string parameter;
    std::string name() const
    {
        return "@param:" + parameter;
    }
    shape compute_shape(std::vector<shape> input) const
    {
        throw "builtin"; 
    }
    argument compute(std::vector<argument> input) const
    {
        throw "builtin";
    }
};

}

} // namespace rtg

#endif
