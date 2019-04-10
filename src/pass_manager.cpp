#include <migraphx/program.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void run_passes(program& prog, const std::vector<pass>& passes, tracer trace)
{
    for(auto& p : passes)
    {
        trace("Pass: ", p.name());
        p.apply(prog);
        trace(prog);

#ifndef NDEBUG
        trace("Validate ...");
        auto invalid = prog.validate();
        if(invalid != prog.end())
        {
            auto index = std::distance(prog.begin(), invalid);
            MIGRAPHX_THROW(p.name() + " pass produces invalid program at instruction " +
                           std::to_string(index) + ": " + invalid->name());
        }
        trace();
#endif
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
