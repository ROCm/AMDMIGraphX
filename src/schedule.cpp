#include <migraphx/schedule.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void schedule::apply(program& p) const
{
    // Compute accumulated weights
    std::unordered_map<instruction_ref, std::size_t> weights;
    auto last = std::prev(p.end());
    fix<std::size_t>([&](auto self, auto ins) -> std::size_t {
        if(weights.count(ins) == 0)
        {
            weights[ins] =
                std::accumulate(ins->inputs().begin(),
                                ins->inputs().end(),
                                model.weight(ins->get_operator()),
                                [&](std::size_t w, instruction_ref i) { return w + self(i); });
        }
        return weights[ins];
    })(last);

    // Topo sort
    fix([&](auto self, auto ins) {
        for(auto i : ins->inputs())
            p.move_instruction(i, p.begin());
        for(auto i : ins->inputs())
            self(i);
    })(last);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
