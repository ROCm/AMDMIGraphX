#ifndef MIGRAPHX_GUARD_RTGLIB_CLONEABLE_HPP
#define MIGRAPHX_GUARD_RTGLIB_CLONEABLE_HPP

#include <migraphx/config.hpp>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <typename Base>
struct cloneable
{
    friend Base;

    virtual std::shared_ptr<Base> clone() = 0;

    template <typename Derived>
    struct derive : Base
    {
        friend Derived;

        std::shared_ptr<Base> clone()
        {
            return std::make_shared<Derived>(static_cast<const Derived&>(*this));
        }
        template <typename... Args>
        derive(Args&&... args) : Base(std::forward<Args>(args)...)
        {
        }
    };

    struct share : Base, std::enable_shared_from_this<Base>
    {
        std::shared_ptr<Base> clone() { return this->shared_from_this(); }
        template <typename... Args>
        share(Args&&... args) : Base(std::forward<Args>(args)...)
        {
        }
    };
    cloneable()                 = default;
    cloneable(const cloneable&) = default;
    cloneable& operator=(const cloneable&) = default;
    virtual ~cloneable() {}
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
