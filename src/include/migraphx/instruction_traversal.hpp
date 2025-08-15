#ifndef MIGRAPHX_GUARD_MIGRAPHX_INSTRUCTION_TRAVERSAL_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_INSTRUCTION_TRAVERSAL_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/unfold.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline auto get_output_path(instruction_ref ins)
{
    return unfold(ins, [](instruction_ref out) -> std::optional<instruction_ref> {
        if(out->outputs().size() != 1)
            return std::nullopt;
        auto next = out->outputs().front();
        return next;
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_INSTRUCTION_TRAVERSAL_HPP
