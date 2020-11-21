#include <migraphx/program.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
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

void run_passes(module& modl, const std::vector<pass>& passes, tracer trace)
{
    for(const auto& p : passes)
    {
        trace("Pass: ", p.name());
        p.apply(modl);
        trace(modl);

#ifndef NDEBUG
        trace("Validate ...");
        auto invalid = modl.validate();
        if(invalid != modl.end())
        {
            auto index = std::distance(modl.begin(), invalid);
            MIGRAPHX_THROW(p.name() + " pass produces invalid program at instruction " +
                           std::to_string(index) + ": " + invalid->name());
        }
        trace();
#endif
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
