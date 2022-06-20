#ifndef MIGRAPHX_GUARD_MIGRAPHX_COMMON_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_COMMON_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct operation;

std::vector<std::size_t> compute_broadcasted_lens(std::vector<std::size_t> s0,
                                                  std::vector<std::size_t> s1);
shape common_shape(const std::vector<shape>& shapes);

instruction_ref insert_common_op(module& m,
                                 instruction_ref ins,
                                 const operation& op,
                                 std::vector<instruction_ref> inputs);
instruction_ref add_common_op(module& m, const operation& op, std::vector<instruction_ref> inputs);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COMMON_HPP
