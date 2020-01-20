#ifndef MIGRAPHX_GUARD_API_RTGLIB_MIGRAPHX_HPP
#define MIGRAPHX_GUARD_API_RTGLIB_MIGRAPHX_HPP

#include <migraphx/migraphx.h>
#include <memory>
#include <exception>
#include <vector>

namespace migraphx {
inline namespace api {

template <class T, class F, class... Ts>
T* make(F f, Ts&&... xs)
{
    T* result = nullptr;
    auto e    = f(&result, std::forward<Ts>(xs)...);
    if(e != migraphx_status_success)
        throw std::runtime_error("Failed to call function");
    return result;
}

template <class F, class... Ts>
void call(F f, Ts&&... xs)
{
    auto e = f(std::forward<Ts>(xs)...);
    if(e != migraphx_status_success)
        throw std::runtime_error("Failed to call function");
}

template <class T, class Deleter, Deleter deleter>
struct handle_base
{
    template <class F, class... Ts>
    void make_handle(F f, Ts&&... xs)
    {
        T* result = nullptr;
        auto e    = f(&result, std::forward<Ts>(xs)...);
        if(e != migraphx_status_success)
            throw std::runtime_error("Failed to call function");
        set_handle(result);
    }

    template <class F, class... Ts>
    void call_handle(F f, Ts&&... xs)
    {
        auto e = f(this->get_handle_ptr(), std::forward<Ts>(xs)...);
        if(e != migraphx_status_success)
            throw std::runtime_error("Failed to call function");
    }

    const std::shared_ptr<T>& get_handle() const { return m_handle; }

    T* get_handle_ptr() const
    {
        assert(m_handle != nullptr);
        return get_handle().get();
    }

    void set_handle(T* ptr, bool own = true)
    {
        if(own)
            m_handle = std::shared_ptr<T>{ptr, deleter};
        else
            m_handle = std::shared_ptr<T>{ptr, [](T*) {}};
    }

    protected:
    std::shared_ptr<T> m_handle;
};

#define MIGRAPHX_HANDLE(name)                                       \
    struct name : handle_base<migraphx_##name,                      \
                              decltype(&migraphx_##name##_destroy), \
                              migraphx_##name##_destroy>

MIGRAPHX_HANDLE(shape){shape() : m_handle(nullptr){}

                       shape(migraphx_shape p, bool own = true) :
                           m_handle(nullptr){this->set_handle(p, own);
} // namespace api

shape(std::vector<size_t> plengths) : m_handle(nullptr)
{
    m_handle = this->make_handle(&migraphx_shape_create, &pshape, plengths);
}

std::vector<size_t> lengths() const
{
    const size_t* pout;
    size_t pout_size;
    this->call_handle(&migraphx_shape_lengths, &pout, &pout_size);
    return std::vector<size_t>(pout, pout + out_size);
}

std::vector<size_t> strides() const
{
    const size_t* pout;
    size_t pout_size;
    this->call_handle(&migraphx_shape_strides, &pout, &pout_size);
    return std::vector<size_t>(pout, pout + out_size);
}

migraphx_shape_datatype_t type() const
{
    migraphx_shape_datatype_t pout;
    this->call_handle(&migraphx_shape_type, &pout);
    return pout;
}
}; // namespace migraphx

MIGRAPHX_HANDLE(argument){argument() :
                              m_handle(nullptr){} argument(migraphx_argument p, bool own = true) :
                                  m_handle(nullptr){this->set_handle(p, own);
}

argument(void* pbuffer) : m_handle(nullptr)
{
    m_handle = this->make_handle(&migraphx_argument_create, &pargument, pbuffer);
}

shape get_shape() const
{
    const_migraphx_shape_t pout;
    this->call_handle(&migraphx_argument_shape, &pout);
    return shape(pout, false);
}

char* data() const
{
    char* pout;
    this->call_handle(&migraphx_argument_buffer, &pout);
    return pout;
}
}
;

MIGRAPHX_HANDLE(target){target() : m_handle(nullptr){} target(migraphx_target p, bool own = true) :
                            m_handle(nullptr){this->set_handle(p, own);
}

target(const char* name) : m_handle(nullptr)
{
    m_handle = this->make_handle(&migraphx_target_create, name);
}
}
;

MIGRAPHX_HANDLE(program_parameter_shapes){
    program_parameter_shapes() : m_handle(nullptr){}

    program_parameter_shapes(migraphx_program_parameter_shapes p, bool own = true) :
        m_handle(nullptr){this->set_handle(p, own);
}

size_t size() const
{
    size_t pout;
    this->call_handle(&migraphx_program_parameter_shapes_size, &pout);
    return pout;
}

shape operator[](const char* pname) const
{
    const_migraphx_shape_t pout;
    this->call_handle(&migraphx_program_parameter_shapes_get, &pout, pname);
    return shape(pout, false);
}

std::vector<const char*> names() const
{
    std::vector<const char*> result(this->size());
    this->call_handle(&migraphx_program_parameter_shapes_names, result.data());
    return result;
}
}
;

MIGRAPHX_HANDLE(program_parameters){
    program_parameters(migraphx_program_parameters p, bool own = true) :
        m_handle(nullptr){this->set_handle(p, own);
}

program_parameters() : m_handle(nullptr)
{
    m_handle = this->make_handle(&migraphx_program_parameters_create, &pprogram_parameters);
}

void add(const char* pname, argument pargument) const
{
    this->call_handle(&migraphx_program_parameters_add, pname, pargument.get_handle_ptr());
}
}
;

MIGRAPHX_HANDLE(program){program() : m_handle(nullptr){}

                         program(migraphx_program p, bool own = true) :
                             m_handle(nullptr){this->set_handle(p, own);
}

void compile(target ptarget, migraphx_compile_options poptions) const
{
    this->call_handle(&migraphx_program_compile, ptarget.get_handle_ptr(), &poptions);
}

void compile(target ptarget) const
{
    this->call_handle(&migraphx_program_compile, ptarget.get_handle_ptr(), nullptr);
}

program_parameter_shapes get_parameter_shapes() const
{
    migraphx_program_parameter_shapes_t pout;
    this->call_handle(&migraphx_program_get_parameter_shapes, &pout);
    return program_parameter_shapes(pout);
}

argument eval(program_parameters pparams) const
{
    migraphx_argument_t pout;
    this->call_handle(&migraphx_program_run, &pout, pparams.get_handle_ptr());
    return argument(pout);
}
}
;

program parse_onnx(const char* filename, migraphx_onnx_options options)
{
    return program(make<migraphx_program>(&migraphx_parse_onnx, name, &options));
}

program parse_onnx(const char* filename)
{
    return program(make<migraphx_program>(&migraphx_parse_onnx, name, nullptr));
}

} // namespace api
} // namespace migraphx

#endif
