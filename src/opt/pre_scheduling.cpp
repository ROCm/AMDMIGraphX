#include <migraphx/pre_scheduling.hpp>
#include "pre_scheduling_impl.hpp"

namespace migraphx {

void pre_scheduling::apply(program& p) const
{
    if(!enabled(MIGRAPHX_DISABLE_PRE_SCHEDULING{}))
    {
        pre_scheduling_impl opt(&p, weight_func, num_of_streams, insert_instr, verify);
        opt.run();
    }
}
} // namespace migraphx
