#ifndef MIGRAPHX_GUARD_MIGRAPHX_ANY_PTR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_ANY_PTR_HPP

#include <migraphx/config.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/type_name.hpp>
#include <cassert>
#include <string_view>
#include <typeindex>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct any_ptr
{
    any_ptr() = default;
    template <class T>
    any_ptr(T* p) : ptr(p), ti(typeid(T*)), name(get_name<T*>())
    {
    }

    any_ptr(void* p, std::string_view pname) : ptr(p), name(pname) {}

    void* get(std::string_view n) const
    {
        if(name != n)
            MIGRAPHX_THROW("any_ptr: type mismatch: " + std::string{name} +
                           " != " + std::string{n});
        return ptr;
    }

    template <class T>
    T get() const
    {
        static_assert(std::is_pointer<T>{}, "Must be a pointer");
        assert(ptr != nullptr);
        if(ti and std::type_index{typeid(T)} != *ti)
            MIGRAPHX_THROW("any_ptr: type mismatch: " + std::string{name} + " != " + get_name<T>());
        else if(name != get_name<T>())
            MIGRAPHX_THROW("any_ptr: type mismatch: " + std::string{name} + " != " + get_name<T>());
        return reinterpret_cast<T>(ptr);
    }
    void* unsafe_get() const { return ptr; }

    private:
    void* ptr                    = nullptr;
    optional<std::type_index> ti = nullopt;
    std::string_view name        = "";

    template <class T>
    static const std::string& get_name()
    {
        return get_type_name<std::remove_cv_t<std::remove_pointer_t<T>>>();
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_ANY_PTR_HPP
