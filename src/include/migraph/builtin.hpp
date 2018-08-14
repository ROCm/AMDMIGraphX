#ifndef MIGRAPH_GUARD_BUILTIN_HPP
#define MIGRAPH_GUARD_BUILTIN_HPP

#include <migraph/context.hpp>
#include <migraph/errors.hpp>
#include <migraph/argument.hpp>

namespace migraph {

namespace builtin {

struct literal
{
    std::string name() const { return "@literal"; }
    shape compute_shape(std::vector<shape>) const { MIGRAPH_THROW("builtin"); }
    argument compute(context&, shape, std::vector<argument>) const { MIGRAPH_THROW("builtin"); }
};

struct outline
{
    shape s;
    std::string name() const { return "@outline"; }
    shape compute_shape(std::vector<shape>) const { return s; }
    argument compute(context&, shape, std::vector<argument>) const { MIGRAPH_THROW("builtin"); }
};

struct param
{
    std::string parameter;
    std::string name() const { return "@param"; }
    shape compute_shape(std::vector<shape>) const { MIGRAPH_THROW("builtin"); }
    argument compute(context&, shape, std::vector<argument>) const { MIGRAPH_THROW("builtin"); }
    friend std::ostream& operator<<(std::ostream& os, const param& op)
    {
        os << op.name() << ":" << op.parameter;
        return os;
    }
};

} // namespace builtin

} // namespace migraph

#endif
