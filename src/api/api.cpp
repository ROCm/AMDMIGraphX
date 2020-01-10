
#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/target.hpp>
#include <migraphx/cpu/target.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/target.hpp>
#endif

#if defined(MIGRAPHX_USE_CLANG_TIDY)
#define MIGRAPHX_OBJECT_CAST reinterpret_cast
#else
#define MIGRAPHX_OBJECT_CAST reinterpret_cast
#endif

#define MIGRAPHX_DEFINE_OBJECT(object, ...)                                \
    inline __VA_ARGS__& migraphx_get_object(object& obj)                   \
    {                                                                    \
        return MIGRAPHX_OBJECT_CAST<__VA_ARGS__&>(obj.handle);                    \
    }                                                                    \
    inline const __VA_ARGS__& migraphx_get_object(const object& obj)       \
    {                                                                    \
        return MIGRAPHX_OBJECT_CAST<const __VA_ARGS__&>(obj.handle);              \
    }

namespace migraphx {

namespace detail {

template <class T>
T& get_object_impl(rank<0>, T& x)
{
    return x;
}

template <class T>
auto get_object_impl(rank<1>, T& x) -> decltype(migraphx_get_object(x))
{
    if(x.handle == nullptr)
        MIGRAPHX_THROW(migraphx_status_bad_param, "Dereferencing nullptr");
    return migraphx_get_object(x);
}

} // namespace detail

template <class T>
auto get_object(T& x) -> decltype(detail::get_object_impl(rank<1>{}, x))
{
    return detail::get_object_impl(rank<1>{}, x);
}

template <class F>
migraphx_status try_(F f, bool output = true)
{
    try
    {
        f();
    }
    catch(const migraphx::exception& ex)
    {
        if(output)
            std::cerr << "MIGraphX Error: " << ex.what() << std::endl;
        if (ex.error > 0)
            return migraphx_status(ex.error);
        else
            return migraphx_status_unknown_error;
    }
    catch(const std::exception& ex)
    {
        if(output)
            std::cerr << "MIGraphX Error: " << ex.what() << std::endl;
        return migraphx_status_unknown_error;
    }
    catch(...)
    {
        return migraphx_status_unknown_error;
    }
    return migraphx_status_success;
}

template <class T>
auto deref(T&& x, migraphx_status err = migraphx_status_bad_param)
    -> decltype((x == nullptr), *x)
{
    if(x == nullptr)
    {
        MIGRAPHX_THROW(err, "Dereferencing nullptr");
    }
    return *x;
}

template<class T>
using object_type = std::decay_t<decltype(get_object(std::declval<T&>()))>;

template<class T>
migraphx_status destroy_object(T& x)
{
    return try_([&] { delete &get_object(x); });
}

template<class R, class... Ts>
migraphx_status create_object(R* r, Ts... xs)
{
    return try_([&] { deref(r).handle = new object_type<R>(xs...); });
}

template<class T>
std::vector<T> to_vector(const T* data, size_t size)
{
    if (size == 0)
        return {};
    if (data == nullptr)
        MIGRAPHX_THROW(migraphx_status_bad_param, "Dereferencing nullptr");
    return {data, data+size};
}

shape::type_t to_shape_type(migraphx_shape_datatype_t t)
{
    switch(t)
    {
#define MIGRAPHX_SHAPE_CASE_CONVERT(x, y) \
    case migraphx_shape_ ## x: return shape::x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

migraphx_shape_datatype_t to_shape_type(shape::type_t t)
{
    switch(t)
    {
#define MIGRAPHX_SHAPE_CASE_CONVERT(x, y) \
    case shape::x: return migraphx_shape_ ## x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

} // namespace migraphx

using migraphx::create_object;
using migraphx::deref;
using migraphx::destroy_object;
using migraphx::get_object;
using migraphx::to_shape_type;
using migraphx::to_vector;
using migraphx::try_;

MIGRAPHX_DEFINE_OBJECT(migraphx_shape, migraphx::shape)

extern "C" migraphx_status migraphx_shape_create(migraphx_shape* shape, 
                                      migraphx_shape_datatype_t type,
                                      const size_t dim_num,
                                      const size_t *dims,
                                      const size_t *strides) 
{
    if (strides)
        return create_object(shape, to_shape_type(type), to_vector(dims, dim_num), to_vector(strides, dim_num));
    else
        return create_object(shape, to_shape_type(type), to_vector(dims, dim_num));
}

extern "C" migraphx_status migraphx_shape_destroy(migraphx_shape shape)
{
    return destroy_object(shape);
}

extern "C" migraphx_status migraphx_shape_get(migraphx_shape shape,
                                      migraphx_shape_datatype_t * type,
                                      size_t * dim_num,
                                      const size_t **dims,
                                      const size_t **strides)
{
    return try_([&] {
        deref(type) = to_shape_type(get_object(shape).type());
        deref(dim_num) = get_object(shape).lens().size();
        deref(dims) = get_object(shape).lens().data();
        deref(strides) = get_object(shape).strides().data();
    });
}


extern "C" migraphx_status migraphx_target_create(migraphx_target* target, 
                                      const char* name)
{
    return try_([&] {
        std::string tname = name;
        migraphx::target t;
        if (tname == "cpu")
            t = migraphx::cpu::target();
#if HAVE_GPU
        else if (tname == "gpu")
            t = migraphx::gpu::target();
#endif
        else
            MIGRAPHX_THROW("Unknown target");
        deref(target).handle = new migraphx::target(t);
    });
}

extern "C" migraphx_status migraphx_target_destroy(migraphx_target target)
{
    return destroy_object(target);
}
