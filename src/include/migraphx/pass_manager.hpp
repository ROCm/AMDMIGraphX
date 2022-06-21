#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_PASS_MANAGER_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_PASS_MANAGER_HPP

#include <migraphx/config.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/tracer.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager
{
    module_pass_manager()                                  = default;
    module_pass_manager(const module_pass_manager&)        = delete;
    virtual module& get_module()                           = 0;
    virtual module* create_module(const std::string& name) = 0;
    virtual void run_pass(const pass& p)                   = 0;

    protected:
    virtual ~module_pass_manager() {}
};

void run_passes(module& mod, const std::vector<pass>& passes, tracer trace = tracer{});
void run_passes(program& prog, const std::vector<pass>& passes, tracer trace = tracer{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
