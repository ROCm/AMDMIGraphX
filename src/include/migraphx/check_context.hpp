#ifndef MIGRAPH_GUARD_RTGLIB_CHECK_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CHECK_CONTEXT_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

template <class T>
struct check_context
{
    struct op
    {
        std::string name() const { return "check_context"; }
        shape compute_shape(const std::vector<shape>&) const { return {}; }
        argument compute(context& ctx, const shape&, const std::vector<argument>&) const
        {
            T* x = any_cast<T>(&ctx);
            if(x == nullptr)
                MIGRAPH_THROW(std::string("Unexpected context type: ") + ctx.type_id().name());
            return {};
        }
    };

    std::string name() const { return "check_context"; }
    void apply(program& p) const { p.insert_instruction(p.begin(), op{}); }
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
