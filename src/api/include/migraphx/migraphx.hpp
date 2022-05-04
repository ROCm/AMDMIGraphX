#ifndef MIGRAPHX_GUARD_API_RTGLIB_MIGRAPHX_HPP
#define MIGRAPHX_GUARD_API_RTGLIB_MIGRAPHX_HPP

#include "migraphx.h"
#include <initializer_list>
#include <migraphx/migraphx.h>
#include <memory>
#include <exception>
#include <vector>
#include <cassert>
#include <iostream>

namespace migraphx {
#ifndef DOXYGEN
inline namespace api { // NOLINT
#endif

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(deprecated)
#define MIGRAPHX_DEPRECATED(...) [[deprecated(__VA_ARGS__)]]
#endif
#endif

#ifndef MIGRAPHX_DEPRECATED
#define MIGRAPHX_DEPRECATED(...)
#endif

template <int N>
struct rank : rank<N - 1>
{
};

template <>
struct rank<0>
{
};

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

template <class F, class Iterator = std::size_t>
struct iota_iterator
{
    Iterator index;
    F f;

    using difference_type   = std::ptrdiff_t;
    using reference         = decltype(f(std::declval<Iterator>()));
    using value_type        = typename std::remove_reference<reference>::type;
    using pointer           = typename std::add_pointer<value_type>::type;
    using iterator_category = std::input_iterator_tag;

    iota_iterator& operator+=(int n)
    {
        index += n;
        return *this;
    }

    iota_iterator& operator-=(int n)
    {
        index += n;
        return *this;
    }

    iota_iterator& operator++()
    {
        index++;
        return *this;
    }

    iota_iterator& operator--()
    {
        index--;
        return *this;
    }

    iota_iterator operator++(int) // NOLINT
    {
        iota_iterator it = *this;
        index++;
        return it;
    }

    iota_iterator operator--(int) // NOLINT
    {
        iota_iterator it = *this;
        index--;
        return it;
    }
    // TODO: operator->
    reference operator*() const { return f(index); }

    friend iota_iterator operator+(iota_iterator x, iota_iterator y)
    {
        return iota_iterator(x.index + y.index, x.f);
    }

    friend iota_iterator operator-(iota_iterator x, iota_iterator y)
    {
        return iota_iterator(x.index - y.index, x.f);
    }

    friend bool operator==(iota_iterator x, iota_iterator y) { return x.index == y.index; }

    friend bool operator!=(iota_iterator x, iota_iterator y) { return x.index != y.index; }
};

template <class Derived>
struct array_base
{
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    template <class T>
    using value_type_t = decltype(std::declval<T>()[0]);

    struct iterator_read
    {
        const Derived* self;
        template <class D = Derived>
        value_type_t<D> operator()(size_t pidx) const
        {
            return (*self)[pidx];
        }
    };

    template <class T>
    using iterator_t = iota_iterator<iterator_read>;

    bool empty() const { return derived().size() == 0; }

    template <class D = Derived>
    value_type_t<D> front() const
    {
        return derived()[0];
    }

    template <class D = Derived>
    value_type_t<D> back() const
    {
        return derived()[derived().size() - 1];
    }

    template <class D = Derived>
    iterator_t<D> begin() const
    {
        return {0, {&derived()}};
    }

    template <class D = Derived>
    iterator_t<D> end() const
    {
        return {derived().size(), {&derived()}};
    }
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

template <class T>
struct holder
{
    // Friend injection
    friend auto migraphx_adl_handle_lookup(holder<T>);
    // Function left unimplemented since its only used in non-evaluated
    // context
    T get() const;
};

template <class C, class T>
struct handle_lookup
{
    friend auto migraphx_adl_handle_lookup(holder<T>) { return holder<C>{}; }
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <class T>
using as_handle = decltype(
    migraphx_adl_handle_lookup(holder<std::remove_cv_t<std::remove_pointer_t<T>>>{}).get());

struct own
{
};
struct borrow
{
};

template <class T>
struct share
{
    share(std::shared_ptr<T> p) : ptr(std::move(p)) {}

    template <class U>
    std::shared_ptr<U> alias(U* p) const
    {
        return std::shared_ptr<U>{ptr, p};
    }

    private:
    std::shared_ptr<T> ptr;
};

template <class Derived, class T, class D, D Deleter, class A, A Assigner>
struct handle_base : handle_lookup<Derived, std::remove_cv_t<T>>
{
    using handle_type = T;
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
        m_handle = std::shared_ptr<U>{ptr, Deleter};
    }

    template <class U>
    void set_handle(U* ptr, borrow)
    {
        m_handle = std::shared_ptr<U>{ptr, [](U*) {}};
    }

    template <class U, class V>
    void set_handle(U* ptr, share<V> b)
    {
        m_handle = std::shared_ptr<T>{ptr, [b](U*) {}};
    }

    share<T> share_handle() const { return {m_handle}; }

    template <class U>
    void assign_to_handle(U* x)
    {
        Assigner(x, this->get_handle_ptr());
    }

    protected:
    std::shared_ptr<T> m_handle;
};

// NOLINTNEXTLINE
#define MIGRAPHX_HANDLE_CONSTRUCTOR(name)                                                          \
    template <class HandleType,                                                                    \
              class Lifetime,                                                                      \
              class =                                                                              \
                  typename std::enable_if<std::is_convertible<HandleType*, handle_type*>{}>::type> \
    name(HandleType* p, Lifetime lifetime)                                                         \
    {                                                                                              \
        this->set_handle(p, std::move(lifetime));                                                  \
    }

template <class Base>
struct interface_base : Base
{
    interface_base() : Base() {}

    protected:
    template <class F>
    static migraphx_status try_(F f) // NOLINT
    {
        try
        {
            f();
            return migraphx_status_success;
        }
        catch(...)
        {
            return migraphx_status_unknown_error;
        }
    }

    template <class F, class T, class... Ts>
    void make_interface(F f, T& obj, Ts&&... xs)
    {
        auto copy = [](void** out, void* input) {
            return try_([&] {
                T** y = reinterpret_cast<T**>(out);
                T* x  = reinterpret_cast<T*>(input);
                assert(x != nullptr and y != nullptr and *y == nullptr);
                // cppcheck-suppress useSmartPointer
                *y = new T(*x); // NOLINT
            });
        };
        auto del = [](void* input) {
            return try_([&] {
                T* x = reinterpret_cast<T*>(input);
                delete x; // NOLINT
            });
        };
        this->make_handle(f, &obj, copy, del, std::forward<Ts>(xs)...);
    }

    template <class T, class Setter, class F>
    void set_fp(Setter setter, F pf)
    {
        static F f = pf;
        (void)f; // avoid warning on gcc
        call(setter, this->get_handle_ptr(), [](auto... xs) -> migraphx_status {
            return try_([&] { call_cast_arg<T>(rank<1>{}, f, xs...); });
        });
    }

    template <class T, class Setter, class F>
    void set_auto_fp(Setter setter, F f)
    {
        return set_fp<T>(setter, [=](T& obj, auto out, auto... xs) {
            auto_invoke(f, out, obj, auto_convert_param(rank<2>{}, xs)...);
        });
    }

    struct no_out_arg
    {
    };

    template <class T, class F, class X, class... Xs, class = std::enable_if_t<std::is_void<X>{}>>
    static void call_cast_arg(rank<0>, F f, X* obj, Xs... xs)
    {
        f(reinterpret_cast<T*>(obj), no_out_arg{}, xs...);
    }

    template <class T,
              class F,
              class R,
              class X,
              class... Xs,
              class = std::enable_if_t<std::is_void<X>{}>>
    static void call_cast_arg(rank<1>, F f, R result, X* obj, Xs... xs)
    {
        f(*reinterpret_cast<T*>(obj), result, xs...);
    }

    template <class F, class T, class... Ts>
    void auto_invoke(F f, T* out, Ts&&... xs)
    {
        auto_assign(rank<2>{}, out, f(std::forward<Ts>(xs)...));
    }

    template <class F, class T, class... Ts>
    void auto_invoke(F f, no_out_arg, Ts&&... xs)
    {
        f(std::forward<Ts>(xs)...);
    }

    template <class T, class = std::enable_if_t<std::is_fundamental<T>{} or std::is_enum<T>{}>>
    T auto_convert_param(rank<0>, T x)
    {
        return x;
    }

    template <class T>
    auto auto_convert_param(rank<1>, T x) -> decltype(as_handle<T>{x})
    {
        return as_handle<T>{x};
    }

    template <class T>
    auto auto_convert_param(rank<2>, T x) -> decltype(as_handle<T>{x, borrow{}})
    {
        return as_handle<T>{x, borrow{}};
    }

    template <class T, class U>
    void auto_assign(rank<0>, T* out, U x)
    {
        return *out = x;
    }

    template <class T, class U>
    auto auto_assign(rank<1>, T* out, U x) -> decltype(x.assign_to_handle(out))
    {
        x.assign_to_handle(out);
    }
};

// NOLINTNEXTLINE
#define MIGRAPHX_INTERFACE_LIFT(T, prefix, name)          \
    this->set_auto_fp<T>(&migraphx_##prefix##_set_##name, \
                         [](T& x, auto... xs) { return x.name(xs...); })

template <class Base, class T>
using require_interface =
    std::enable_if_t<std::is_base_of<Base, T>{} and not std::is_same<T, Base>{} and
                     std::is_copy_constructible<T>{} and std::is_final<T>{}>;

#ifdef DOXYGEN
#define MIGRAPHX_DETAIL_HANDLE_BASE(name, const_) handle_base<>
#else
#define MIGRAPHX_DETAIL_HANDLE_BASE(name, const_)       \
    handle_base<name,                                   \
                const_ migraphx_##name,                 \
                decltype(&migraphx_##name##_destroy),   \
                migraphx_##name##_destroy,              \
                decltype(&migraphx_##name##_assign_to), \
                migraphx_##name##_assign_to>
#endif
// NOLINTNEXTLINE
#define MIGRAPHX_HANDLE_BASE(name) MIGRAPHX_DETAIL_HANDLE_BASE(name, )
// NOLINTNEXTLINE
#define MIGRAPHX_CONST_HANDLE_BASE(name) MIGRAPHX_DETAIL_HANDLE_BASE(name, const)

/**
 * @brief Describe shape of tensor
 * @details A shape consists of a data type, lengths of multi-dimension tensor, and strides
 *
 */
struct shape : MIGRAPHX_CONST_HANDLE_BASE(shape)
{
    shape() {}

    MIGRAPHX_HANDLE_CONSTRUCTOR(shape);

    /// Construct a scalar shape
    shape(migraphx_shape_datatype_t type)
    {
        this->make_handle(&migraphx_shape_create_scalar, type);
    }

    /// Construct a shape with its type and lengths. The strides are
    /// automatically computed assumming a packed layout.
    shape(migraphx_shape_datatype_t type, std::vector<size_t> plengths)
    {
        this->make_handle(&migraphx_shape_create, type, plengths.data(), plengths.size());
    }

    shape(migraphx_shape_datatype_t type,
          std::vector<size_t> plengths,
          std::vector<size_t> pstrides)
    {
        this->make_handle(&migraphx_shape_create_with_strides,
                          type,
                          plengths.data(),
                          plengths.size(),
                          pstrides.data(),
                          pstrides.size());
    }

    std::vector<size_t> lengths() const
    {
        const size_t* pout;
        size_t pout_size;
        call(&migraphx_shape_lengths, &pout, &pout_size, this->get_handle_ptr());
        return {pout, pout + pout_size};
    }

    std::vector<size_t> strides() const
    {
        const size_t* pout;
        size_t pout_size;
        call(&migraphx_shape_strides, &pout, &pout_size, this->get_handle_ptr());
        return {pout, pout + pout_size};
    }

    migraphx_shape_datatype_t type() const
    {
        migraphx_shape_datatype_t pout;
        call(&migraphx_shape_type, &pout, this->get_handle_ptr());
        return pout;
    }

    size_t bytes() const
    {
        size_t pout;
        call(&migraphx_shape_bytes, &pout, this->get_handle_ptr());
        return pout;
    }

    friend bool operator==(const shape& px, const shape& py)
    {
        bool pout;
        call(&migraphx_shape_equal, &pout, px.get_handle_ptr(), py.get_handle_ptr());
        return pout;
    }

    friend bool operator!=(const shape& px, const shape& py) { return !(px == py); }
};

/**
 * @brief Arguments to be passed to an migraphx arguments
 *
 * An `argument` represents a raw buffer of data with a shape.
 *
 */
struct argument : MIGRAPHX_CONST_HANDLE_BASE(argument)
{
    argument() {}

    MIGRAPHX_HANDLE_CONSTRUCTOR(argument);

    MIGRAPHX_DEPRECATED("Contructor without lifetime annotation is deprecated.")
    argument(const migraphx_argument* p) { this->set_handle(p, borrow{}); }

    argument(shape pshape, void* pbuffer)
    {
        this->make_handle(&migraphx_argument_create, pshape.get_handle_ptr(), pbuffer);
    }

    shape get_shape() const
    {
        const_migraphx_shape_t pout;
        call(&migraphx_argument_shape, &pout, this->get_handle_ptr());
        return {pout, this->share_handle()};
    }

    char* data() const
    {
        char* pout;
        call(&migraphx_argument_buffer, &pout, this->get_handle_ptr());
        return pout;
    }

    /// Generate an argument using random data
    static argument generate(shape ps, size_t pseed = 0)
    {
        return {make<migraphx_argument>(&migraphx_argument_generate, ps.get_handle_ptr(), pseed),
                own{}};
    }

    friend bool operator==(const argument& px, const argument& py)
    {
        bool pout;
        call(&migraphx_argument_equal, &pout, px.get_handle_ptr(), py.get_handle_ptr());
        return pout;
    }

    friend bool operator!=(const argument& px, const argument& py) { return !(px == py); }
};

/// A target for compilation
struct target : MIGRAPHX_HANDLE_BASE(target)
{
    target() {}

    MIGRAPHX_HANDLE_CONSTRUCTOR(target);

    /// Construct a target from its name
    target(const char* name) { this->make_handle(&migraphx_target_create, name); }
};

struct program_parameter_shapes : MIGRAPHX_HANDLE_BASE(program_parameter_shapes)
{
    program_parameter_shapes() {}

    MIGRAPHX_HANDLE_CONSTRUCTOR(program_parameter_shapes);

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
        return {pout, this->share_handle()};
    }

    std::vector<const char*> names() const
    {
        std::vector<const char*> result(this->size());
        if(!result.empty())
        {
            call(&migraphx_program_parameter_shapes_names, result.data(), this->get_handle_ptr());
        }
        return result;
    }
};

/// A class to construct the inputs parameters for a program
struct program_parameters : MIGRAPHX_HANDLE_BASE(program_parameters)
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(program_parameters);

    MIGRAPHX_DEPRECATED("Contructor without lifetime annotation is deprecated.")
    program_parameters(migraphx_program_parameters* p) { this->set_handle(p, borrow{}); }

    program_parameters() { this->make_handle(&migraphx_program_parameters_create); }

    /// Construct the parameters from initializer_list
    program_parameters(std::initializer_list<std::pair<std::string, argument>> l)
    {
        this->make_handle(&migraphx_program_parameters_create);
        for(auto&& p : l)
            this->add(p.first.c_str(), p.second);
    }

    /// Add a new parameter
    void add(const char* pname, const argument& pargument) const
    {
        call(&migraphx_program_parameters_add,
             this->get_handle_ptr(),
             pname,
             pargument.get_handle_ptr());
    }
};

struct arguments : MIGRAPHX_HANDLE_BASE(arguments), array_base<arguments>
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(arguments);

    size_t size() const
    {
        size_t pout;
        call(&migraphx_arguments_size, &pout, this->get_handle_ptr());
        return pout;
    }

    argument operator[](size_t pidx) const
    {
        const_migraphx_argument_t pout;
        call(&migraphx_arguments_get, &pout, this->get_handle_ptr(), pidx);
        return {pout, this->share_handle()};
    }
};

struct shapes : MIGRAPHX_HANDLE_BASE(shapes), array_base<shapes>
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(shapes);

    size_t size() const
    {
        size_t pout;
        call(&migraphx_shapes_size, &pout, this->get_handle_ptr());
        return pout;
    }

    shape operator[](size_t pidx) const
    {
        const_migraphx_shape_t pout;
        call(&migraphx_shapes_get, &pout, this->get_handle_ptr(), pidx);
        return {pout, this->share_handle()};
    }
};

struct operation : MIGRAPHX_HANDLE_BASE(operation)
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(operation);

    template <class... Ts>
    operation(const char* name, const char* attributes = nullptr, Ts... xs)
    {
        this->make_handle(&migraphx_operation_create, name, attributes, xs...);
    }

    std::string name()
    {
        std::array<char, 1024> out_name;
        call(&migraphx_operation_name, out_name.data(), 1024, this->get_handle_ptr());
        return {out_name.data()};
    }
};

struct instruction : MIGRAPHX_CONST_HANDLE_BASE(instruction)
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(instruction);
};

struct instructions : MIGRAPHX_HANDLE_BASE(instructions)
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(instructions);

    template <class... Ts>
    instructions(Ts... xs)
    {
        std::array<const_migraphx_instruction_t, sizeof...(Ts)> a{xs.get_handle_ptr()...};
        this->make_handle(&migraphx_instructions_create, a.data(), a.size());
    }
};

struct module;

struct modules : MIGRAPHX_HANDLE_BASE(modules)
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(modules);

    template <class... Ts>
    modules(Ts... xs)
    {
        std::array<migraphx_module_t, sizeof...(Ts)> a = {xs.get_handle_ptr()...};
        this->make_handle(&migraphx_modules_create, a.data(), a.size());
    }
};

struct module
{
    MIGRAPHX_DEPRECATED("Constructor without lifetime annotation is deprecated.")
    module(migraphx_module* m) : mm(std::shared_ptr<migraphx_module*>(), m) {}

    module(migraphx_module* m, borrow) : mm(std::shared_ptr<migraphx_module*>(), m) {}

    template <class T>
    module(migraphx_module* m, share<T> b) : mm(b.alias(m))
    {
    }

    void print() const { call(&migraphx_module_print, mm.get()); }

    instruction add_instruction(const migraphx::operation& op, const migraphx::instructions& args)
    {
        migraphx_instruction_t op_ins;
        call(&migraphx_module_add_instruction,
             &op_ins,
             mm.get(),
             op.get_handle_ptr(),
             args.get_handle_ptr());
        return instruction(op_ins, own{});
    }

    instruction add_instruction(const migraphx::operation& op,
                                const migraphx::instructions& args,
                                const migraphx::modules& module_args)
    {
        migraphx_instruction_t op_ins;
        call(&migraphx_module_add_instruction_with_mod_args,
             &op_ins,
             mm.get(),
             op.get_handle_ptr(),
             args.get_handle_ptr(),
             module_args.get_handle_ptr());
        return instruction(op_ins, own{});
    }

    template <typename T>
    instruction add_literal(const migraphx::shape& s, T* buffer)
    {
        migraphx_instruction_t literal_ins;
        const auto* buffer_ptr = reinterpret_cast<const char*>(buffer);
        call(&migraphx_module_add_literal, &literal_ins, mm.get(), s.get_handle_ptr(), buffer_ptr);
        return instruction(literal_ins, own{});
    }

    instruction add_parameter(const std::string& name, shape s)
    {
        migraphx_instruction_t param_ins;
        call(
            &migraphx_module_add_parameter, &param_ins, mm.get(), name.c_str(), s.get_handle_ptr());
        return instruction(param_ins, own{});
    }

    instruction add_return(const migraphx::instructions& args)
    {
        migraphx_instruction_t ret_ins;
        call(&migraphx_module_add_return, &ret_ins, mm.get(), args.get_handle_ptr());
        return instruction(ret_ins, own{});
    }

    migraphx_module_t get_handle_ptr() const { return mm.get(); }

    private:
    std::shared_ptr<migraphx_module> mm;
};

struct context : MIGRAPHX_HANDLE_BASE(context)
{
    context() { this->make_handle(&migraphx_context_create); }

    MIGRAPHX_HANDLE_CONSTRUCTOR(context);

    void finish() const { call(&migraphx_context_finish, this->get_handle_ptr()); }

    template <class T>
    T get_queue()
    {
        void* out;
        call(&migraphx_context_get_queue, &out, this->get_handle_ptr());
        // TODO: check type here
        return reinterpret_cast<T>(out);
    }

    private:
    std::shared_ptr<migraphx_context> ctx;
};

struct compile_options : MIGRAPHX_HANDLE_BASE(compile_options)
{
    compile_options() { this->make_handle(&migraphx_compile_options_create); }

    MIGRAPHX_HANDLE_CONSTRUCTOR(compile_options);

    /// For targets with offloaded memory(such as the gpu), this will insert
    /// instructions during compilation to copy the input parameters to the
    /// offloaded memory and to copy the final result from the offloaded
    /// memory back to main memory.
    void set_offload_copy(bool value = true)
    {
        call(&migraphx_compile_options_set_offload_copy, this->get_handle_ptr(), value);
    }

    /// Optimize math functions to use faster approximate versions. There may
    /// be slight accuracy degredation when enabled.
    void set_fast_math(bool value = true)
    {
        call(&migraphx_compile_options_set_fast_math, this->get_handle_ptr(), value);
    }
};

/// A program represents the all computation graphs to be compiled and executed
struct program : MIGRAPHX_HANDLE_BASE(program)
{
    program() { this->make_handle(&migraphx_program_create); }

    MIGRAPHX_HANDLE_CONSTRUCTOR(program);

    /// Compile the program for a specific target to be ran on
    void compile(const target& ptarget, const compile_options& poptions) const
    {
        call(&migraphx_program_compile,
             this->get_handle_ptr(),
             ptarget.get_handle_ptr(),
             poptions.get_handle_ptr());
    }

    /// Compile the program for a specific target to be ran on
    void compile(const target& ptarget) const
    {
        call(&migraphx_program_compile,
             this->get_handle_ptr(),
             ptarget.get_handle_ptr(),
             migraphx::compile_options{}.get_handle_ptr());
    }

    /// Return the shapes for the input parameters
    program_parameter_shapes get_parameter_shapes() const
    {
        migraphx_program_parameter_shapes_t pout;
        call(&migraphx_program_get_parameter_shapes, &pout, this->get_handle_ptr());
        return program_parameter_shapes(pout, own{});
    }

    /// Get the shapes of all the outputs returned by this program
    shapes get_output_shapes() const
    {
        migraphx_shapes_t pout;
        call(&migraphx_program_get_output_shapes, &pout, this->get_handle_ptr());
        return shapes(pout, own{});
    }

    /// Run the program using the inputs passed in
    arguments eval(const program_parameters& pparams) const
    {
        migraphx_arguments_t pout;
        call(&migraphx_program_run, &pout, this->get_handle_ptr(), pparams.get_handle_ptr());
        return arguments(pout, own{});
    }

    void print() const { call(&migraphx_program_print, this->get_handle_ptr()); }

    program sort()
    {
        call(&migraphx_program_sort, this->get_handle_ptr());
        return *this;
    }

    friend bool operator==(const program& px, const program& py)
    {
        bool pout;
        call(&migraphx_program_equal, &pout, px.get_handle_ptr(), py.get_handle_ptr());
        return pout;
    }

    module get_main_module()
    {
        migraphx_module_t p_modu;
        call(&migraphx_program_get_main_module, &p_modu, this->get_handle_ptr());
        return module{p_modu, this->share_handle()};
    }

    context experimental_get_context()
    {
        migraphx_context_t ctx;
        call(&migraphx_program_experimental_get_context, &ctx, this->get_handle_ptr());
        return context{ctx, this->share_handle()};
    }

    module create_module(const std::string& name)
    {
        migraphx_module_t p_modu;
        call(&migraphx_program_create_module, &p_modu, this->get_handle_ptr(), name.data());
        return module{p_modu, this->share_handle()};
    }

    friend bool operator!=(const program& px, const program& py) { return !(px == py); }
};

// options for migraphx file format options
struct file_options : MIGRAPHX_HANDLE_BASE(file_options)
{
    MIGRAPHX_HANDLE_CONSTRUCTOR(file_options);
    file_options() { this->make_handle(&migraphx_file_options_create); }

    // set file format
    void set_file_format(const char* format)
    {
        call(&migraphx_file_options_set_file_format, this->get_handle_ptr(), format);
    }
};

/// Load a saved migraphx program from a file
inline program load(const char* filename, const file_options& options)
{
    return program(make<migraphx_program>(&migraphx_load, filename, options.get_handle_ptr()),
                   own{});
}

/// Load a saved migraphx program from a file
inline program load(const char* filename)
{
    return program(
        make<migraphx_program>(&migraphx_load, filename, migraphx::file_options{}.get_handle_ptr()),
        own{});
}

/// Save a program to a file
inline void save(const program& p, const char* filename, const file_options& options)
{
    call(&migraphx_save, p.get_handle_ptr(), filename, options.get_handle_ptr());
}

/// Save a program to a file
inline void save(const program& p, const char* filename)
{
    call(&migraphx_save, p.get_handle_ptr(), filename, migraphx::file_options{}.get_handle_ptr());
}

/// Options for parsing onnx options
struct onnx_options : MIGRAPHX_HANDLE_BASE(onnx_options)
{
    onnx_options() { this->make_handle(&migraphx_onnx_options_create); }

    MIGRAPHX_HANDLE_CONSTRUCTOR(onnx_options);

    /// Make onnx parser treat an inputs with a certain dimensions
    void set_input_parameter_shape(const std::string& name, std::vector<std::size_t> dim)
    {
        call(&migraphx_onnx_options_set_input_parameter_shape,
             this->get_handle_ptr(),
             name.c_str(),
             dim.data(),
             dim.size());
    }

    /// When there is a dimension parameter, then use this default value
    void set_default_dim_value(unsigned int value)
    {
        call(&migraphx_onnx_options_set_default_dim_value, this->get_handle_ptr(), value);
    }

    /// Set default max iteration number for the loop operator
    void set_default_loop_iterations(int64_t value)
    {
        call(&migraphx_onnx_options_set_default_loop_iterations, this->get_handle_ptr(), value);
    }
};

/// Parse an onnx file into a migraphx program
inline program parse_onnx(const char* filename, const migraphx::onnx_options& options)
{
    return program(make<migraphx_program>(&migraphx_parse_onnx, filename, options.get_handle_ptr()),
                   own{});
}

/// Parse an onnx file into a migraphx program
inline program parse_onnx(const char* filename)
{
    migraphx::onnx_options options;
    return program(make<migraphx_program>(&migraphx_parse_onnx, filename, options.get_handle_ptr()),
                   own{});
}

/// Parse a buffer of memory as an onnx file
inline program
parse_onnx_buffer(const void* data, size_t size, const migraphx::onnx_options& options)
{
    return program(
        make<migraphx_program>(&migraphx_parse_onnx_buffer, data, size, options.get_handle_ptr()),
        own{});
}

/// Parse a buffer of memory as an onnx file
inline program parse_onnx_buffer(const void* data, size_t size)
{
    migraphx::onnx_options options;
    return program(
        make<migraphx_program>(&migraphx_parse_onnx_buffer, data, size, options.get_handle_ptr()),
        own{});
}

/// Parse a buffer of memory as an onnx file
inline program parse_onnx_buffer(const std::string& buffer, const migraphx::onnx_options& options)
{
    return program(
        make<migraphx_program>(
            &migraphx_parse_onnx_buffer, buffer.data(), buffer.size(), options.get_handle_ptr()),
        own{});
}

/// Parse a buffer of memory as an onnx file
inline program parse_onnx_buffer(const std::string& buffer)
{
    migraphx::onnx_options options;
    return program(
        make<migraphx_program>(
            &migraphx_parse_onnx_buffer, buffer.data(), buffer.size(), options.get_handle_ptr()),
        own{});
}

/// Options for parsing tf options
struct tf_options : MIGRAPHX_HANDLE_BASE(tf_options)
{
    tf_options() { this->make_handle(&migraphx_tf_options_create); }

    MIGRAPHX_HANDLE_CONSTRUCTOR(tf_options);

    /// Make tf parser treat an inputs with a certain dimensions
    void set_input_parameter_shape(const std::string& name, std::vector<std::size_t> dim)
    {
        call(&migraphx_tf_options_set_input_parameter_shape,
             this->get_handle_ptr(),
             name.c_str(),
             dim.data(),
             dim.size());
    }

    /// Change data layout to NHWC (default is NCHW)
    void set_nhwc(bool is_nhwc = true)
    {
        call(&migraphx_tf_options_set_nhwc, this->get_handle_ptr(), is_nhwc);
    }

    /// When there is a dimension parameter, then use this default value
    void set_default_dim_value(unsigned int value)
    {
        call(&migraphx_tf_options_set_default_dim_value, this->get_handle_ptr(), value);
    }

    /// Set output node names to return specific outputs from graph
    void set_output_names(std::vector<const char*> names)
    {
        call(&migraphx_tf_options_set_output_names,
             this->get_handle_ptr(),
             names.data(),
             names.size());
    }
};

/// Parse a tf file into a migraphx program
inline program parse_tf(const char* filename, const migraphx::tf_options& options)
{
    return program(make<migraphx_program>(&migraphx_parse_tf, filename, options.get_handle_ptr()),
                   own{});
}

/// Parse a tf file into a migraphx program
inline program parse_tf(const char* filename)
{
    migraphx::tf_options options;
    return program(make<migraphx_program>(&migraphx_parse_tf, filename, options.get_handle_ptr()),
                   own{});
}

struct quantize_op_names : MIGRAPHX_HANDLE_BASE(quantize_op_names)
{
    quantize_op_names() { this->make_handle(&migraphx_quantize_op_names_create); }

    MIGRAPHX_HANDLE_CONSTRUCTOR(quantize_op_names);

    void add(const std::string& name)
    {
        call(&migraphx_quantize_op_names_add, this->get_handle_ptr(), name.c_str());
    }
};

/// Quantize program to use fp16
inline void quantize_fp16(const program& prog, const quantize_op_names& names)
{
    call(&migraphx_quantize_fp16_with_op_names, prog.get_handle_ptr(), names.get_handle_ptr());
}

/// Quantize program to use fp16
inline void quantize_fp16(const program& prog)
{
    call(&migraphx_quantize_fp16, prog.get_handle_ptr());
}

/// Options to be passed when quantizing for int8
struct quantize_int8_options : MIGRAPHX_HANDLE_BASE(quantize_int8_options)
{
    quantize_int8_options() { this->make_handle(&migraphx_quantize_int8_options_create); }

    MIGRAPHX_HANDLE_CONSTRUCTOR(quantize_int8_options);

    /// Add an operator that should be quantized
    void add_op_name(const std::string& name)
    {
        call(&migraphx_quantize_int8_options_add_op_name, this->get_handle_ptr(), name.c_str());
    }

    /// Add calibrartion data to be used for quantizing
    void add_calibration_data(const program_parameters& pp)
    {
        call(&migraphx_quantize_int8_options_add_calibration_data,
             this->get_handle_ptr(),
             pp.get_handle_ptr());
    }
};

/// Quantize program to use int8
inline void
quantize_int8(const program& prog, const target& ptarget, const quantize_int8_options& options)
{
    call(&migraphx_quantize_int8,
         prog.get_handle_ptr(),
         ptarget.get_handle_ptr(),
         options.get_handle_ptr());
}

struct experimental_custom_op_base
{
    virtual std::string name() const                                            = 0;
    virtual argument compute(context ctx, shape output, arguments inputs) const = 0;
    virtual shape compute_shape(shapes inputs) const                            = 0;
    virtual ~experimental_custom_op_base()                                      = default;
};

struct experimental_custom_op : interface_base<MIGRAPHX_HANDLE_BASE(experimental_custom_op)>
{
    template <class T>
    experimental_custom_op(T& obj)
    {
        this->make_interface(&migraphx_experimental_custom_op_create, obj, obj.name().c_str());
        MIGRAPHX_INTERFACE_LIFT(T, experimental_custom_op, compute_shape);
        MIGRAPHX_INTERFACE_LIFT(T, experimental_custom_op, compute);
    }

    void register_op() { call(&migraphx_experimental_custom_op_register, this->get_handle_ptr()); }
};

template <class T, class = require_interface<experimental_custom_op_base, T>>
void register_experimental_custom_op(T& obj)
{
    experimental_custom_op op{obj};
    op.register_op();
}

#ifndef DOXYGEN
} // namespace api
#endif
} // namespace migraphx

#endif
