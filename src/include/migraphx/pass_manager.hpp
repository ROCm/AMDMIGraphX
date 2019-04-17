#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_PASS_MANAGER_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_PASS_MANAGER_HPP

#include <list>
#include <unordered_map>
#include <migraphx/operation.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/target.hpp>
#include <migraphx/tracer.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>
#include <algorithm>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void run_passes(program& prog, const std::vector<pass>& passes, tracer trace = tracer{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
