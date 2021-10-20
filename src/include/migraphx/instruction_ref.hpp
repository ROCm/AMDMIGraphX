#ifndef MIGRAPHX_GUARD_INSTRUCTION_REF_HPP
#define MIGRAPHX_GUARD_INSTRUCTION_REF_HPP

#include <list>
#include <functional>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct instruction;
using instruction_ref = std::list<instruction>::iterator;

migraphx::instruction* as_address(const instruction_ref& ins) noexcept;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {
template <>
struct hash<migraphx::instruction_ref>
{
    using argument_type = migraphx::instruction_ref;
    using result_type   = std::size_t;
    result_type operator()(const migraphx::instruction_ref& x) const noexcept
    {
        return std::hash<migraphx::instruction*>{}(migraphx::as_address(x));
    }
};

template <>
struct equal_to<migraphx::instruction_ref>
{
    using argument_type = migraphx::instruction_ref;
    using result_type   = bool;
    result_type operator()(const migraphx::instruction_ref& x,
                           const migraphx::instruction_ref& y) const noexcept
    {
        return migraphx::as_address(x) == migraphx::as_address(y);
    }
};

} // namespace std

#endif
