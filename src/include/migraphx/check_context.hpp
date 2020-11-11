#ifndef MIGRAPHX_GUARD_RTGLIB_CHECK_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CHECK_CONTEXT_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
struct check_context
{
    struct op : auto_register_op<op>
    {
        std::string name() const { return "check_context::" + get_type_name<T>(); }
        shape compute_shape(const std::vector<shape>&) const { return {}; }
        argument compute(context& ctx, const shape&, const std::vector<argument>&) const
        {
            this->check(ctx);
            return {};
        }
        void finalize(context& ctx, const shape&, const std::vector<shape>&) const
        {
            this->check(ctx);
        }
        void check(context& ctx) const
        {
            T* x = any_cast<T>(&ctx);
            if(x == nullptr)
                MIGRAPHX_THROW(std::string("Unexpected context type: ") + ctx.type_id().name());
        }
    };

    std::string name() const { return "check_context"; }
    void apply(module& p) const { p.insert_instruction(p.begin(), op{}); }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
