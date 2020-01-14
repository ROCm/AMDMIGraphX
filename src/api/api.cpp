
#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
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

#define MIGRAPHX_DEFINE_OBJECT(object, ...)                          \
    inline __VA_ARGS__& migraphx_get_object(object& obj)             \
    {                                                                \
        return MIGRAPHX_OBJECT_CAST<__VA_ARGS__&>(obj.handle);       \
    }                                                                \
    inline const __VA_ARGS__& migraphx_get_object(const object& obj) \
    {                                                                \
        return MIGRAPHX_OBJECT_CAST<const __VA_ARGS__&>(obj.handle); \
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
        if(ex.error > 0)
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
auto deref(T&& x, migraphx_status err = migraphx_status_bad_param) -> decltype((x == nullptr), *x)
{
    if(x == nullptr)
    {
        MIGRAPHX_THROW(err, "Dereferencing nullptr");
    }
    return *x;
}

template <class T>
using object_type = std::decay_t<decltype(get_object(std::declval<T&>()))>;

template <class T>
migraphx_status destroy_object(T& x)
{
    return try_([&] { delete &get_object(x); });
}

template <class R, class... Ts>
migraphx_status allocate_object(R* r, Ts... xs)
{
    return try_([&] { deref(r).handle = new object_type<R>(xs...); });
}

template<class T, class F>
migraphx_status create_object(T* x, F f)
{
    return try_([&] {
        deref(x).handle = new object_type<T>(f());
    });
}

template<class T, class F>
migraphx_status set_object(T* x, F f)
{
    return try_([&] {
        deref(x).handle = std::addressof(f());
    });
}

template <class T>
std::vector<T> to_vector(const T* data, size_t size)
{
    if(size == 0)
        return {};
    if(data == nullptr)
        MIGRAPHX_THROW(migraphx_status_bad_param, "Dereferencing nullptr");
    return {data, data + size};
}

shape::type_t to_shape_type(migraphx_shape_datatype_t t)
{
    switch(t)
    {
#define MIGRAPHX_SHAPE_CASE_CONVERT(x, y) \
    case migraphx_shape_##x: return shape::x;
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
    case shape::x: return migraphx_shape_##x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

} // namespace migraphx

using migraphx::allocate_object;
using migraphx::deref;
using migraphx::destroy_object;
using migraphx::get_object;
using migraphx::create_object;
using migraphx::set_object;
using migraphx::to_shape_type;
using migraphx::to_vector;
using migraphx::try_;

MIGRAPHX_DEFINE_OBJECT(migraphx_shape, migraphx::shape)

extern "C" migraphx_status migraphx_shape_create(migraphx_shape* shape,
                                                 migraphx_shape_datatype_t type,
                                                 const size_t dim_num,
                                                 const size_t* dims,
                                                 const size_t* strides)
{
    if(strides)
        return allocate_object(
            shape, to_shape_type(type), to_vector(dims, dim_num), to_vector(strides, dim_num));
    else
        return allocate_object(shape, to_shape_type(type), to_vector(dims, dim_num));
}

extern "C" migraphx_status migraphx_shape_destroy(migraphx_shape shape)
{
    return destroy_object(shape);
}

extern "C" migraphx_status migraphx_shape_get(migraphx_shape shape,
                                              migraphx_shape_datatype_t* type,
                                              size_t* dim_num,
                                              const size_t** dims,
                                              const size_t** strides)
{
    return try_([&] {
        deref(type)    = to_shape_type(get_object(shape).type());
        deref(dim_num) = get_object(shape).lens().size();
        deref(dims)    = get_object(shape).lens().data();
        deref(strides) = get_object(shape).strides().data();
    });
}

MIGRAPHX_DEFINE_OBJECT(migraphx_argument, migraphx::argument)

extern "C" migraphx_status migraphx_argument_create(migraphx_argument* argument, migraphx_shape shape, void *buffer)
{
    return allocate_object(argument, get_object(shape), buffer);
}

extern "C" migraphx_status migraphx_argument_destroy(migraphx_argument argument)
{
    return destroy_object(argument);
}

MIGRAPHX_DEFINE_OBJECT(migraphx_target, migraphx::target)

extern "C" migraphx_status migraphx_target_create(migraphx_target* target, const char* name)
{
    return try_([&] {
        std::string tname = name;
        migraphx::target t;
        if(tname == "cpu")
            t = migraphx::cpu::target();
#ifdef HAVE_GPU
        else if(tname == "gpu")
            t = migraphx::gpu::target();
#endif
        else
            MIGRAPHX_THROW(migraphx_status_unknown_target, "Unknown target");
        deref(target).handle = new migraphx::target(t);
    });
}

extern "C" migraphx_status migraphx_target_destroy(migraphx_target target)
{
    return destroy_object(target);
}

extern "C" migraphx_status migraphx_target_copy_to(migraphx_target target, migraphx_argument src, migraphx_argument* dst)
{
    return create_object(dst, [&] {
        return get_object(target).copy_to(get_object(src));
    });
}

extern "C" migraphx_status migraphx_target_copy_from(migraphx_target target, migraphx_argument src, migraphx_argument* dst)
{
    return create_object(dst, [&] {
        return get_object(target).copy_from(get_object(src));
    });
}

MIGRAPHX_DEFINE_OBJECT(migraphx_program, migraphx::program)

extern "C" migraphx_status migraphx_program_create(migraphx_program* program)
{
    return allocate_object(program);
}

extern "C" migraphx_status migraphx_program_destroy(migraphx_program program)
{
    return destroy_object(program);
}

extern "C" migraphx_status migraphx_program_compile(migraphx_program program, migraphx_target target, migraphx_compile_options* options)
{
    return try_([&] {
        migraphx::compile_options o{};
        if (options)
        {
            o.offload_copy = options->offload_copy;
        }
        get_object(program).compile(get_object(target), o);
    });
}

MIGRAPHX_DEFINE_OBJECT(migraphx_program_parameter_shapes, std::unordered_map<std::string, migraphx::shape>)

extern "C" migraphx_status migraphx_program_parameter_shapes_create(migraphx_program_parameter_shapes* program_parameter_shapes, migraphx_program program)
{
    return create_object(program_parameter_shapes, [&] {
        return get_object(program).get_parameter_shapes();
    });
}

extern "C" migraphx_status migraphx_program_parameter_shapes_destroy(migraphx_program_parameter_shapes program_parameter_shapes)
{
    return destroy_object(program_parameter_shapes);
}

extern "C" migraphx_status migraphx_program_parameter_shapes_size(migraphx_program_parameter_shapes program_parameter_shapes, size_t * size)
{
    return try_([&] {
        deref(size) = get_object(program_parameter_shapes).size();
    });
}

extern "C" migraphx_status migraphx_program_parameter_shapes_names(migraphx_program_parameter_shapes program_parameter_shapes, const char ** names)
{
    return try_([&] {
        if (not names)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Derefencing null pointer");
        int i = 0;
        for(auto&& p:get_object(program_parameter_shapes))
        {
            names[i] = p.first.c_str();
            i++;
        }
    });
}

extern "C" migraphx_status migraphx_program_parameter_shapes_get_shape(migraphx_program_parameter_shapes program_parameter_shapes, const char * name, migraphx_shape* shape)
{
    return set_object(shape, [&]() -> decltype(auto) {
        return get_object(program_parameter_shapes).at(name);
    });
}

MIGRAPHX_DEFINE_OBJECT(migraphx_program_parameters, migraphx::program::parameter_map)

extern "C" migraphx_status migraphx_program_parameters_create(migraphx_program_parameters* program_parameters)
{
    return allocate_object(program_parameters);
}

extern "C" migraphx_status migraphx_program_parameters_destroy(migraphx_program_parameters program_parameters)
{
    return destroy_object(program_parameters);
}

extern "C" migraphx_status migraphx_program_parameters_add(migraphx_program_parameters program_parameters, const char* name, migraphx_argument argument)
{
    return try_([&] {
        get_object(program_parameters)[name] = get_object(argument);
    });
}

extern "C" migraphx_status migraphx_program_run(migraphx_program program, migraphx_program_parameters program_parameters, migraphx_argument* output)
{
    return create_object(output, [&] {
        return get_object(program).eval(get_object(program_parameters));
    });
}
