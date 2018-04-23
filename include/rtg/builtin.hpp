#ifndef RTG_GUARD_BUILTIN_HPP
#define RTG_GUARD_BUILTIN_HPP

#include <rtg/operand.hpp>

namespace rtg {

namespace builtin {

struct literal
{
    std::string name() const { return "@literal"; }
    shape compute_shape(std::vector<shape>) const { throw "builtin"; }
    argument compute(std::vector<argument>) const { throw "builtin"; }
};

struct param
{
    std::string parameter;
    std::string name() const { return "@param:" + parameter; }
    shape compute_shape(std::vector<shape>) const { throw "builtin"; }
    argument compute(std::vector<argument>) const { throw "builtin"; }
};

} // namespace builtin

} // namespace rtg

#endif
