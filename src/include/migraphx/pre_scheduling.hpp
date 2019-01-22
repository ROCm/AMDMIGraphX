#ifndef MIGRAPHX_GUARD_RTGLIB_PRE_SCHEDULING_HPP
#define MIGRAPHX_GUARD_RTGLIB_PRE_SCHEDULING_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>

#include <migraphx/program.hpp>

namespace migraphx {

struct pre_scheduling
{
    std::function<std::pair<int, int>(const operation&)> weight_func;
    int num_of_streams;
    std::string name() const { return "pre scheduling"; }
    void apply(program& p) const;
};
} // namespace migraphx

#endif
