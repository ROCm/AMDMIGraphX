#ifndef MIGRAPHX_GUARD_API_RTGLIB_MIGRAPHX_HPP
#define MIGRAPHX_GUARD_API_RTGLIB_MIGRAPHX_HPP

#include <migraphx/migraphx.h>
#include <memory>
#include <exception>
#include <vector>
#include <cassert>

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

struct own
{
};
struct borrow
{
};

template <class T, class Deleter, Deleter deleter>
struct handle_base
{
    handle_base() : m_handle(nullptr) {}
    template <class F, class... Ts>
    void make_handle(F f, Ts&&... xs)
    {
        using type = typename std::remove_cv<T>::type;
        set_handle(make<type>(f, std::forward<Ts>(xs)...), own{});
    }

    const std::shared_ptr<T>& get_handle() const { return m_handle; }

    T* get_handle_ptr() const
    {
        assert(m_handle != nullptr);
        return get_handle().get();
    }

    template <class U>
    void set_handle(U* ptr, own)
    {
        m_handle = std::shared_ptr<U>{ptr, deleter};
    }

    template <class U>
    void set_handle(U* ptr, borrow)
    {
        m_handle = std::shared_ptr<U>{ptr, [](U*) {}};
    }

    protected:
    std::shared_ptr<T> m_handle;
};

#define MIGRAPHX_HANDLE_BASE(name, const_)            \
    handle_base<const_ migraphx_##name,               \
                decltype(&migraphx_##name##_destroy), \
                migraphx_##name##_destroy>

#define MIGRAPHX_HANDLE(name) struct name : MIGRAPHX_HANDLE_BASE(name, )

#define MIGRAPHX_CONST_HANDLE(name) struct name : MIGRAPHX_HANDLE_BASE(name, const)

// clang-format off
MIGRAPHX_CONST_HANDLE(shape)
{
    shape() {}

    shape(const migraphx_shape* p)
    {
        this->set_handle(p, borrow{});
    }

    shape(migraphx_shape* p, own)
    {
        this->set_handle(p, own{});
    }

    shape(migraphx_shape* p, borrow)
    {
        this->set_handle(p, borrow{});
    }

    shape(migraphx_shape_datatype_t type, std::vector<size_t> plengths)
    {
        this->make_handle(&migraphx_shape_create, type, plengths.data(), plengths.size());
    }

    std::vector<size_t> lengths() const
    {
        const size_t* pout;
        size_t pout_size;
        call(&migraphx_shape_lengths, &pout, &pout_size, this->get_handle_ptr());
        return std::vector<size_t>(pout, pout + pout_size);
    }

    std::vector<size_t> strides() const
    {
        const size_t* pout;
        size_t pout_size;
        call(&migraphx_shape_strides, &pout, &pout_size, this->get_handle_ptr());
        return std::vector<size_t>(pout, pout + pout_size);
    }

    migraphx_shape_datatype_t type() const
    {
        migraphx_shape_datatype_t pout;
        call(&migraphx_shape_type, &pout, this->get_handle_ptr());
        return pout;
    }
};

MIGRAPHX_HANDLE(argument)
{
    argument() {}

    argument(migraphx_argument * p, borrow)
    {
        this->set_handle(p, borrow{});
    }

    argument(migraphx_argument* p, own)
    {
        this->set_handle(p, own{});
    }

    argument(shape pshape, void* pbuffer)
    {
        this->make_handle(&migraphx_argument_create, pshape.get_handle_ptr(), pbuffer);
    }

    shape get_shape() const
    {
        const_migraphx_shape_t pout;
        call(&migraphx_argument_shape, &pout, this->get_handle_ptr());
        return shape(pout);
    }

    char* data() const
    {
        char* pout;
        call(&migraphx_argument_buffer, &pout, this->get_handle_ptr());
        return pout;
    }
};

MIGRAPHX_HANDLE(target)
{
    target() {}

    target(migraphx_target * p, own)
    {
        this->set_handle(p, own{});
    }

    target(migraphx_target* p, borrow)
    {
        this->set_handle(p, borrow{});
    }

    target(const char* name)
    {
        this->make_handle(&migraphx_target_create, name);
    }
};

MIGRAPHX_HANDLE(program_parameter_shapes)
{
    program_parameter_shapes() {}

    program_parameter_shapes(migraphx_program_parameter_shapes * p, own)
    {
        this->set_handle(p, own{});
    }

    program_parameter_shapes(migraphx_program_parameter_shapes* p, borrow)
    {
        this->set_handle(p, borrow{});
    }

    size_t size() const
    {
        size_t pout;
        call(&migraphx_program_parameter_shapes_size, &pout, this->get_handle_ptr());
        return pout;
    }

    shape operator[](const char* pname) const
    {
        const_migraphx_shape_t pout;
        call(&migraphx_program_parameter_shapes_get, &pout, this->get_handle_ptr(), pname);
        return shape(pout);
    }

    std::vector<const char*> names() const
    {
        std::vector<const char*> result(this->size());
        call(&migraphx_program_parameter_shapes_names, result.data(), this->get_handle_ptr());
        return result;
    }
};

MIGRAPHX_HANDLE(program_parameters)
{
    program_parameters(migraphx_program_parameters * p, own)
    {
        this->set_handle(p, own{});
    }

    program_parameters(migraphx_program_parameters* p, borrow)
    {
        this->set_handle(p, borrow{});
    }

    program_parameters()
    {
        this->make_handle(&migraphx_program_parameters_create);
    }

    void add(const char* pname, argument pargument) const
    {
        call(&migraphx_program_parameters_add,
             this->get_handle_ptr(),
             pname,
             pargument.get_handle_ptr());
    }
};

MIGRAPHX_HANDLE(program)
{
    program() {}

    program(migraphx_program * p, own)
    {
        this->set_handle(p, own{});
    }

    program(migraphx_program* p, borrow)
    {
        this->set_handle(p, borrow{});
    }

    void compile(target ptarget, migraphx_compile_options poptions) const
    {
        call(&migraphx_program_compile, this->get_handle_ptr(), ptarget.get_handle_ptr(), &poptions);
    }

    void compile(target ptarget) const
    {
        call(&migraphx_program_compile, this->get_handle_ptr(), ptarget.get_handle_ptr(), nullptr);
    }

    program_parameter_shapes get_parameter_shapes() const
    {
        migraphx_program_parameter_shapes_t pout;
        call(&migraphx_program_get_parameter_shapes, &pout, this->get_handle_ptr());
        return program_parameter_shapes(pout, own{});
    }

    argument eval(program_parameters pparams) const
    {
        migraphx_argument_t pout;
        call(&migraphx_program_run, &pout, this->get_handle_ptr(), pparams.get_handle_ptr());
        return argument(pout, own{});
    }
};
// clang-format on

program parse_onnx(const char* filename, migraphx_onnx_options options)
{
    return program(make<migraphx_program>(&migraphx_parse_onnx, filename, &options), own{});
}

program parse_onnx(const char* filename)
{
    return program(make<migraphx_program>(&migraphx_parse_onnx, filename, nullptr), own{});
}

} // namespace api
} // namespace migraphx

#endif
