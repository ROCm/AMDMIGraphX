#ifndef MIGRAPH_GUARD_RTGLIB_AUTO_ANY_CAST_HPP
#define MIGRAPH_GUARD_RTGLIB_AUTO_ANY_CAST_HPP

namespace migraph {

namespace detail {

template<class U>
void any_cast() {}

template<class T>
struct auto_any_caster
{
    T& x;

    template <class U>
    operator U&()
    {
        return any_cast<U>(x);
    }

    operator T&()
    {
        return x;
    }
};

}

template<class T>
detail::auto_any_caster<T> auto_any_cast(T& x)
{
    return {x};
}

} // namespace migraph

#endif
