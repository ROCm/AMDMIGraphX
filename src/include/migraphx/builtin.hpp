#ifndef MIGRAPHX_GUARD_BUILTIN_HPP
#define MIGRAPHX_GUARD_BUILTIN_HPP

#include <migraphx/context.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace builtin {

struct literal
{
    std::string name() const { return "@literal"; }
    shape compute_shape(const std::vector<shape>&) const { MIGRAPHX_THROW("builtin"); }
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        MIGRAPHX_THROW("builtin");
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
        MIGRAPHX_THROW("builtin");
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
    shape compute_shape(const std::vector<shape>&) const { MIGRAPHX_THROW("builtin"); }
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        MIGRAPHX_THROW("builtin");
    }
    friend std::ostream& operator<<(std::ostream& os, const param& op)
    {
        os << op.name() << ":" << op.parameter;
        return os;
    }
};

struct add_return
{
    std::vector<std::string> output_names;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_names, "output_names"));
    }

    std::string name() const { return "add_return"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }
    argument compute(context&, const shape&, const std::vector<argument>&) const { return {}; }
};

} // namespace builtin
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
