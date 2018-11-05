#ifndef MIGRAPH_GUARD_BUILTIN_HPP
#define MIGRAPH_GUARD_BUILTIN_HPP

#include <migraph/context.hpp>
#include <migraph/errors.hpp>
#include <migraph/argument.hpp>
#include <migraph/reflect.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

namespace builtin {

struct literal
{
    std::string name() const { return "@literal"; }
    shape compute_shape(const std::vector<shape>&) const { MIGRAPH_THROW("builtin"); }
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        MIGRAPH_THROW("builtin");
    }
};

struct outline
{
    shape s;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"));
    }

    std::string name() const { return "@outline"; }
    shape compute_shape(const std::vector<shape>&) const { return s; }
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        MIGRAPH_THROW("builtin");
    }
};

struct param
{
    std::string parameter;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.parameter, "parameter"));
    }

    std::string name() const { return "@param"; }
    shape compute_shape(const std::vector<shape>&) const { MIGRAPH_THROW("builtin"); }
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        MIGRAPH_THROW("builtin");
    }
    friend std::ostream& operator<<(std::ostream& os, const param& op)
    {
        os << op.name() << ":" << op.parameter;
        return os;
    }
};

} // namespace builtin
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
