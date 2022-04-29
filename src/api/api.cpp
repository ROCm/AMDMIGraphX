#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/json.hpp>
#include <migraphx/convert_to_json.hpp>
#include <algorithm>
#include <cstdarg>

namespace migraphx {

template <class F>
migraphx_status try_(F f, bool output = true) // NOLINT
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

shape::type_t to_shape_type(migraphx_shape_datatype_t t)
{
    switch(t)
    {
    case migraphx_shape_tuple_type: return shape::tuple_type;
#define MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT(x, y) \
    case migraphx_shape_##x: return shape::x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

migraphx_shape_datatype_t to_shape_type(shape::type_t t)
{
    switch(t)
    {
    case shape::tuple_type: return migraphx_shape_tuple_type;
#define MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT(x, y) \
    case shape::x: return migraphx_shape_##x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

template <class T>
auto to_obj_vector(const T* x, std::size_t n)
{
    std::vector<decltype((*x)->object)> result;
    std::transform(x, x + n, std::back_inserter(result), [&](auto&& y) { return y->object; });
    return result;
}

template <class T, class U>
auto to_objptr_vector(const U* x, std::size_t n)
{
    std::vector<T> result;
    std::transform(
        x, x + n, std::back_inserter(result), [&](auto&& y) { return std::addressof(y->object); });
    return result;
}

target get_target(const std::string& name) { return make_target(name); }

void set_offload_copy(compile_options& options, bool value) { options.offload_copy = value; }

void set_fast_math(compile_options& options, bool value) { options.fast_math = value; }

void set_file_format(file_options& options, const char* format) { options.format = format; }

void set_default_dim_value(onnx_options& options, size_t value)
{
    options.default_dim_value = value;
}

void set_default_loop_iterations(onnx_options& options, int64_t value)
{
    options.max_loop_iterations = value;
}

void set_nhwc(tf_options& options, bool is_nhwc) { options.is_nhwc = is_nhwc; }

void set_default_dim_value(tf_options& options, size_t value) { options.batch_size = value; }

void set_input_parameter_shape(onnx_options& options,
                               const char* name,
                               std::vector<std::size_t> dims)
{
    options.map_input_dims[std::string(name)] = std::move(dims);
}

void set_input_parameter_shape(tf_options& options, const char* name, std::vector<std::size_t> dims)
{
    options.map_input_dims[std::string(name)] = std::move(dims);
}

void set_output_names(tf_options& options, std::vector<const char*> names)
{
    options.output_node_names = std::vector<std::string>(names.begin(), names.end());
}

template <class Value>
std::vector<const char*> get_names(const std::unordered_map<std::string, Value>& m)
{
    std::vector<const char*> result;
    std::transform(
        m.begin(), m.end(), std::back_inserter(result), [](auto&& p) { return p.first.c_str(); });
    return result;
}

void quantize_fp16_with_op_names(program& prog, std::vector<std::string>& names)
{
    if(names.empty())
    {
        names = {"all"};
    }

    migraphx::quantize_fp16(prog, names);
}

struct quantize_int8_options
{
    std::vector<parameter_map> calibration = {};
    std::vector<std::string> op_names      = {};
};

void add_op_name(quantize_int8_options& options, const char* name)
{
    options.op_names.push_back(name);
}

void add_calibration_data(quantize_int8_options& options, parameter_map& data)
{
    options.calibration.push_back(data);
}

void quantize_int8_wrap(program& prog, const target& t, quantize_int8_options& options)
{
    if(options.op_names.empty())
    {
        options.op_names = {"dot", "convolution"};
    }

    migraphx::quantize_int8(prog, t, options.calibration, options.op_names);
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif

operation create_op(const char* name, const char* attributes, va_list vlist)
{
    std::string sattributes = attributes == nullptr ? "" : attributes;
    std::vector<char> buffer(sattributes.size() * 2);
    std::vsnprintf(buffer.data(), buffer.size(), sattributes.c_str(), vlist);
    value v = value::object{};
    if(attributes != nullptr)
    {
        v = from_json_string(convert_to_json(std::string(buffer.data())));
    }
    auto op = make_op(name, v);

    return op;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template <class T>
bool equal(const T& x, const T& y)
{
    return x == y;
}

std::vector<argument> run(program& p, const parameter_map& params) { return p.eval(params); }

std::vector<shape> get_output_shapes(program& p) { return p.get_output_shapes(); }

void print_program(const program& p) { std::cout << p << std::endl; }

void print_module(const module& m) { std::cout << m << std::endl; }

struct experimental_custom_op
{
    std::string name;
    experimental_custom_op() = default;

    experimental_custom_op(std::string pname) : name(std::move(pname)) {}
};

template <class CustomOp>
struct custom_operation
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }
    CustomOp op;
    std::string name() const { return op.xobject.name; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        return op.compute_shape(std::move(inputs));
    }

    argument compute(const std::vector<argument>&) const { MIGRAPHX_THROW("Not computable"); }
};

template <class CustomOp>
void register_custom_op(const CustomOp& op)
{
    register_op(custom_operation<CustomOp>{op});
}

migraphx::context get_context(const program& p) { return p.get_context(); }

} // namespace migraphx

template <class T, class U, class Target = std::remove_pointer_t<T>>
Target* object_cast(U* x)
{
    return reinterpret_cast<Target*>(x);
}
template <class T, class U, class Target = std::remove_pointer_t<T>>
const Target* object_cast(const U* x)
{
    return reinterpret_cast<const Target*>(x);
}

template <class T, class... Ts, class Target = std::remove_pointer_t<T>>
Target* allocate(Ts&&... xs)
{
    return new Target(std::forward<Ts>(xs)...); // NOLINT
}

template <class T>
void destroy(T* x)
{
    delete x; // NOLINT
}
// TODO: Move to interface preamble
template <class C, class D>
struct manage_generic_ptr
{
    manage_generic_ptr() = default;

    manage_generic_ptr(std::nullptr_t) {}

    manage_generic_ptr(void* pdata, C pcopier, D pdeleter)
        : data(nullptr), copier(pcopier), deleter(pdeleter)
    {
        copier(&data, pdata);
    }

    manage_generic_ptr(const manage_generic_ptr& rhs)
        : data(nullptr), copier(rhs.copier), deleter(rhs.deleter)
    {
        if(copier)
            copier(&data, rhs.data);
    }

    manage_generic_ptr(manage_generic_ptr&& other) noexcept
        : data(other.data), copier(other.copier), deleter(other.deleter)
    {
        other.data    = nullptr;
        other.copier  = nullptr;
        other.deleter = nullptr;
    }

    manage_generic_ptr& operator=(manage_generic_ptr rhs)
    {
        std::swap(data, rhs.data);
        std::swap(copier, rhs.copier);
        std::swap(deleter, rhs.deleter);
        return *this;
    }

    ~manage_generic_ptr()
    {
        if(data != nullptr)
            deleter(data);
    }

    void* data = nullptr;
    C copier   = nullptr;
    D deleter  = nullptr;
};

extern "C" struct migraphx_shape;
struct migraphx_shape
{
    template <class... Ts>
    migraphx_shape(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::shape object;
};

extern "C" struct migraphx_argument;
struct migraphx_argument
{
    template <class... Ts>
    migraphx_argument(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::argument object;
};

extern "C" struct migraphx_target;
struct migraphx_target
{
    template <class... Ts>
    migraphx_target(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::target object;
};

extern "C" struct migraphx_program_parameter_shapes;
struct migraphx_program_parameter_shapes
{
    template <class... Ts>
    migraphx_program_parameter_shapes(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    std::unordered_map<std::string, migraphx::shape> object;
};

extern "C" struct migraphx_program_parameters;
struct migraphx_program_parameters
{
    template <class... Ts>
    migraphx_program_parameters(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    std::unordered_map<std::string, migraphx::argument> object;
};

extern "C" struct migraphx_arguments;
struct migraphx_arguments
{
    template <class... Ts>
    migraphx_arguments(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    std::vector<migraphx::argument> object;
};

extern "C" struct migraphx_shapes;
struct migraphx_shapes
{
    template <class... Ts>
    migraphx_shapes(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    std::vector<migraphx::shape> object;
};

extern "C" struct migraphx_instruction;
struct migraphx_instruction
{
    template <class... Ts>
    migraphx_instruction(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::instruction_ref object;
};

extern "C" struct migraphx_instructions;
struct migraphx_instructions
{
    template <class... Ts>
    migraphx_instructions(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    std::vector<migraphx::instruction_ref> object;
};

extern "C" struct migraphx_modules;
struct migraphx_modules
{
    template <class... Ts>
    migraphx_modules(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    std::vector<migraphx::module*> object;
};

extern "C" struct migraphx_module;
struct migraphx_module
{
    template <class... Ts>
    migraphx_module(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::module object;
};

extern "C" struct migraphx_program;
struct migraphx_program
{
    template <class... Ts>
    migraphx_program(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::program object;
};

extern "C" struct migraphx_operation;
struct migraphx_operation
{
    template <class... Ts>
    migraphx_operation(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::operation object;
};

extern "C" struct migraphx_onnx_options;
struct migraphx_onnx_options
{
    template <class... Ts>
    migraphx_onnx_options(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::onnx_options object;
};

extern "C" struct migraphx_file_options;
struct migraphx_file_options
{
    template <class... Ts>
    migraphx_file_options(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::file_options object;
};

extern "C" struct migraphx_compile_options;
struct migraphx_compile_options
{
    template <class... Ts>
    migraphx_compile_options(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::compile_options object;
};

extern "C" struct migraphx_tf_options;
struct migraphx_tf_options
{
    template <class... Ts>
    migraphx_tf_options(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::tf_options object;
};

extern "C" struct migraphx_quantize_op_names;
struct migraphx_quantize_op_names
{
    template <class... Ts>
    migraphx_quantize_op_names(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    std::vector<std::string> object;
};

extern "C" struct migraphx_quantize_int8_options;
struct migraphx_quantize_int8_options
{
    template <class... Ts>
    migraphx_quantize_int8_options(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::quantize_int8_options object;
};

extern "C" struct migraphx_context;
struct migraphx_context
{
    template <class... Ts>
    migraphx_context(Ts&&... xs)
        : object(std::forward<Ts>(xs)...) // NOLINT(readability-redundant-member-init)
    {
    }
    migraphx::context object;
};

extern "C" struct migraphx_experimental_custom_op;
struct migraphx_experimental_custom_op
{
    template <class... Ts>
    migraphx_experimental_custom_op(void* p,
                                    migraphx_experimental_custom_op_copy c,
                                    migraphx_experimental_custom_op_delete d,
                                    Ts&&... xs)
        : object_ptr(p, c, d), xobject(std::forward<Ts>(xs)...)
    {
    }
    manage_generic_ptr<migraphx_experimental_custom_op_copy, migraphx_experimental_custom_op_delete>
        object_ptr = nullptr;
    migraphx::experimental_custom_op xobject;
    migraphx_experimental_custom_op_compute compute_f = nullptr;
    migraphx::argument compute(migraphx::context ctx,
                               migraphx::shape output,
                               std::vector<migraphx::argument> inputs) const
    {
        std::remove_pointer_t<migraphx_argument_t> out;
        if(compute_f == nullptr)
            throw std::runtime_error("compute function is missing.");
        auto api_error_result = compute_f(&out,
                                          object_ptr.data,
                                          object_cast<migraphx_context_t>(&(ctx)),
                                          object_cast<migraphx_shape_t>(&(output)),
                                          object_cast<migraphx_arguments_t>(&(inputs)));
        if(api_error_result != migraphx_status_success)
            throw std::runtime_error("Error in compute.");
        return (&out)->object;
    }

    migraphx_experimental_custom_op_compute_shape compute_shape_f = nullptr;
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        std::remove_pointer_t<migraphx_shape_t> out;
        if(compute_shape_f == nullptr)
            throw std::runtime_error("compute_shape function is missing.");
        auto api_error_result =
            compute_shape_f(&out, object_ptr.data, object_cast<migraphx_shapes_t>(&(inputs)));
        if(api_error_result != migraphx_status_success)
            throw std::runtime_error("Error in compute_shape.");
        return (&out)->object;
    }
};

extern "C" migraphx_status migraphx_shape_destroy(migraphx_shape_t shape)
{
    auto api_error_result = migraphx::try_([&] { destroy((shape)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shape_assign_to(migraphx_shape_t output,
                                                    const_migraphx_shape_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shape_create(migraphx_shape_t* shape,
                                                 migraphx_shape_datatype_t type,
                                                 size_t* lengths,
                                                 size_t lengths_size)
{
    auto api_error_result = migraphx::try_([&] {
        if(lengths == nullptr and lengths_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter lengths: Null pointer");
        *shape = object_cast<migraphx_shape_t>(
            allocate<migraphx::shape>((migraphx::to_shape_type(type)),
                                      (std::vector<size_t>(lengths, lengths + lengths_size))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shape_create_with_strides(migraphx_shape_t* shape,
                                                              migraphx_shape_datatype_t type,
                                                              size_t* lengths,
                                                              size_t lengths_size,
                                                              size_t* strides,
                                                              size_t strides_size)
{
    auto api_error_result = migraphx::try_([&] {
        if(lengths == nullptr and lengths_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter lengths: Null pointer");
        if(strides == nullptr and strides_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter strides: Null pointer");
        *shape = object_cast<migraphx_shape_t>(
            allocate<migraphx::shape>((migraphx::to_shape_type(type)),
                                      (std::vector<size_t>(lengths, lengths + lengths_size)),
                                      (std::vector<size_t>(strides, strides + strides_size))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shape_create_scalar(migraphx_shape_t* shape,
                                                        migraphx_shape_datatype_t type)
{
    auto api_error_result = migraphx::try_([&] {
        *shape = object_cast<migraphx_shape_t>(
            allocate<migraphx::shape>((migraphx::to_shape_type(type))));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_shape_lengths(const size_t** out, size_t* out_size, const_migraphx_shape_t shape)
{
    auto api_error_result = migraphx::try_([&] {
        if(out == nullptr or out_size == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        auto&& api_result = (shape->object).lens();
        *out              = api_result.data();
        *out_size         = api_result.size();
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_shape_strides(const size_t** out, size_t* out_size, const_migraphx_shape_t shape)
{
    auto api_error_result = migraphx::try_([&] {
        if(out == nullptr or out_size == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        auto&& api_result = (shape->object).strides();
        *out              = api_result.data();
        *out_size         = api_result.size();
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shape_type(migraphx_shape_datatype_t* out,
                                               const_migraphx_shape_t shape)
{
    auto api_error_result = migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = migraphx::to_shape_type((shape->object).type());
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shape_bytes(size_t* out, const_migraphx_shape_t shape)
{
    auto api_error_result = migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = (shape->object).bytes();
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_shape_equal(bool* out, const_migraphx_shape_t shape, const_migraphx_shape_t x)
{
    auto api_error_result = migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        if(x == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter x: Null pointer");
        *out = migraphx::equal((shape->object), (x->object));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_argument_destroy(migraphx_argument_t argument)
{
    auto api_error_result = migraphx::try_([&] { destroy((argument)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_argument_assign_to(migraphx_argument_t output,
                                                       const_migraphx_argument_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_argument_create(migraphx_argument_t* argument, const_migraphx_shape_t shape, void* buffer)
{
    auto api_error_result = migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *argument = object_cast<migraphx_argument_t>(
            allocate<migraphx::argument>((shape->object), (buffer)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_argument_shape(const_migraphx_shape_t* out,
                                                   const_migraphx_argument_t argument)
{
    auto api_error_result = migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        *out = object_cast<const_migraphx_shape_t>(&((argument->object).get_shape()));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_argument_buffer(char** out, const_migraphx_argument_t argument)
{
    auto api_error_result = migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        *out = (argument->object).data();
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_argument_equal(bool* out, const_migraphx_argument_t argument, const_migraphx_argument_t x)
{
    auto api_error_result = migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        if(x == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter x: Null pointer");
        *out = migraphx::equal((argument->object), (x->object));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_argument_generate(migraphx_argument_t* out, const_migraphx_shape_t s, size_t seed)
{
    auto api_error_result = migraphx::try_([&] {
        if(s == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter s: Null pointer");
        *out = allocate<migraphx_argument_t>(migraphx::generate_argument((s->object), (seed)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_target_destroy(migraphx_target_t target)
{
    auto api_error_result = migraphx::try_([&] { destroy((target)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_target_assign_to(migraphx_target_t output,
                                                     const_migraphx_target_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_target_create(migraphx_target_t* target, const char* name)
{
    auto api_error_result = migraphx::try_([&] {
        *target = object_cast<migraphx_target_t>(
            allocate<migraphx::target>(migraphx::get_target((name))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_parameter_shapes_destroy(
    migraphx_program_parameter_shapes_t program_parameter_shapes)
{
    auto api_error_result = migraphx::try_([&] { destroy((program_parameter_shapes)); });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_assign_to(migraphx_program_parameter_shapes_t output,
                                            const_migraphx_program_parameter_shapes_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_size(size_t* out,
                                       migraphx_program_parameter_shapes_t program_parameter_shapes)
{
    auto api_error_result = migraphx::try_([&] {
        if(program_parameter_shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        *out = (program_parameter_shapes->object).size();
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_get(const_migraphx_shape_t* out,
                                      migraphx_program_parameter_shapes_t program_parameter_shapes,
                                      const char* name)
{
    auto api_error_result = migraphx::try_([&] {
        if(program_parameter_shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        *out =
            object_cast<const_migraphx_shape_t>(&((program_parameter_shapes->object).at((name))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_parameter_shapes_names(
    const char** out, migraphx_program_parameter_shapes_t program_parameter_shapes)
{
    auto api_error_result = migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(program_parameter_shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        auto&& api_result = migraphx::get_names((program_parameter_shapes->object));
        std::copy(api_result.begin(), api_result.end(), out);
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_parameters_destroy(migraphx_program_parameters_t program_parameters)
{
    auto api_error_result = migraphx::try_([&] { destroy((program_parameters)); });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_parameters_assign_to(migraphx_program_parameters_t output,
                                      const_migraphx_program_parameters_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_parameters_create(migraphx_program_parameters_t* program_parameters)
{
    auto api_error_result = migraphx::try_([&] {
        *program_parameters = object_cast<migraphx_program_parameters_t>(
            allocate<std::unordered_map<std::string, migraphx::argument>>());
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_parameters_add(migraphx_program_parameters_t program_parameters,
                                const char* name,
                                const_migraphx_argument_t argument)
{
    auto api_error_result = migraphx::try_([&] {
        if(program_parameters == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameters: Null pointer");
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        (program_parameters->object)[(name)] = (argument->object);
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_arguments_destroy(migraphx_arguments_t arguments)
{
    auto api_error_result = migraphx::try_([&] { destroy((arguments)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_arguments_assign_to(migraphx_arguments_t output,
                                                        const_migraphx_arguments_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_arguments_size(size_t* out, migraphx_arguments_t arguments)
{
    auto api_error_result = migraphx::try_([&] {
        if(arguments == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter arguments: Null pointer");
        *out = (arguments->object).size();
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_arguments_get(const_migraphx_argument_t* out, migraphx_arguments_t arguments, size_t idx)
{
    auto api_error_result = migraphx::try_([&] {
        if(arguments == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter arguments: Null pointer");
        *out = object_cast<const_migraphx_argument_t>(&((arguments->object).at((idx))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shapes_destroy(migraphx_shapes_t shapes)
{
    auto api_error_result = migraphx::try_([&] { destroy((shapes)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shapes_assign_to(migraphx_shapes_t output,
                                                     const_migraphx_shapes_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_shapes_size(size_t* out, migraphx_shapes_t shapes)
{
    auto api_error_result = migraphx::try_([&] {
        if(shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shapes: Null pointer");
        *out = (shapes->object).size();
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_shapes_get(const_migraphx_shape_t* out, migraphx_shapes_t shapes, size_t idx)
{
    auto api_error_result = migraphx::try_([&] {
        if(shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shapes: Null pointer");
        *out = object_cast<const_migraphx_shape_t>(&((shapes->object).at((idx))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_instruction_destroy(migraphx_instruction_t instruction)
{
    auto api_error_result = migraphx::try_([&] { destroy((instruction)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_instruction_assign_to(migraphx_instruction_t output,
                                                          const_migraphx_instruction_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_instructions_destroy(migraphx_instructions_t instructions)
{
    auto api_error_result = migraphx::try_([&] { destroy((instructions)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_instructions_assign_to(migraphx_instructions_t output,
                                                           const_migraphx_instructions_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_instructions_create(migraphx_instructions_t* instructions,
                                                        const_migraphx_instruction_t* ptr,
                                                        size_t size)
{
    auto api_error_result = migraphx::try_([&] {
        *instructions =
            object_cast<migraphx_instructions_t>(allocate<std::vector<migraphx::instruction_ref>>(
                migraphx::to_obj_vector<const_migraphx_instruction_t>((ptr), (size))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_modules_destroy(migraphx_modules_t modules)
{
    auto api_error_result = migraphx::try_([&] { destroy((modules)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_modules_assign_to(migraphx_modules_t output,
                                                      const_migraphx_modules_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_modules_create(migraphx_modules_t* modules, migraphx_module_t* ptr, size_t size)
{
    auto api_error_result = migraphx::try_([&] {
        *modules = object_cast<migraphx_modules_t>(allocate<std::vector<migraphx::module*>>(
            migraphx::to_objptr_vector<migraphx::module*>((ptr), (size))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_module_create(migraphx_module_t* module, char* name)
{
    auto api_error_result = migraphx::try_([&] {
        if(name == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter name: Null pointer");
        *module = object_cast<migraphx_module_t>(allocate<migraphx::module>((std::string(name))));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_module_print(const_migraphx_module_t module)
{
    auto api_error_result = migraphx::try_([&] {
        if(module == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module: Null pointer");
        migraphx::print_module((module->object));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_module_add_instruction(migraphx_instruction_t* out,
                                                           migraphx_module_t module,
                                                           migraphx_operation_t op,
                                                           migraphx_instructions_t args)
{
    auto api_error_result = migraphx::try_([&] {
        if(module == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module: Null pointer");
        if(op == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter op: Null pointer");
        if(args == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter args: Null pointer");
        *out = allocate<migraphx_instruction_t>(
            (module->object).add_instruction((op->object), (args->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_module_add_instruction_with_mod_args(migraphx_instruction_t* out,
                                              migraphx_module_t module,
                                              migraphx_operation_t op,
                                              migraphx_instructions_t args,
                                              migraphx_modules_t module_refs)
{
    auto api_error_result = migraphx::try_([&] {
        if(module == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module: Null pointer");
        if(op == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter op: Null pointer");
        if(args == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter args: Null pointer");
        if(module_refs == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module_refs: Null pointer");
        *out = allocate<migraphx_instruction_t>(
            (module->object).add_instruction((op->object), (args->object), (module_refs->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_module_add_literal(migraphx_instruction_t* out,
                                                       migraphx_module_t module,
                                                       const_migraphx_shape_t shape,
                                                       const char* buffer)
{
    auto api_error_result = migraphx::try_([&] {
        if(module == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = allocate<migraphx_instruction_t>(
            (module->object).add_literal((shape->object), (buffer)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_module_add_parameter(migraphx_instruction_t* out,
                                                         migraphx_module_t module,
                                                         const char* name,
                                                         const_migraphx_shape_t shape)
{
    auto api_error_result = migraphx::try_([&] {
        if(module == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = allocate<migraphx_instruction_t>(
            (module->object).add_parameter((name), (shape->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_module_add_return(migraphx_instruction_t* out,
                                                      migraphx_module_t module,
                                                      migraphx_instructions_t args)
{
    auto api_error_result = migraphx::try_([&] {
        if(module == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module: Null pointer");
        if(args == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter args: Null pointer");
        *out = allocate<migraphx_instruction_t>((module->object).add_return((args->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_destroy(migraphx_program_t program)
{
    auto api_error_result = migraphx::try_([&] { destroy((program)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_assign_to(migraphx_program_t output,
                                                      const_migraphx_program_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_create(migraphx_program_t* program)
{
    auto api_error_result = migraphx::try_(
        [&] { *program = object_cast<migraphx_program_t>(allocate<migraphx::program>()); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_get_main_module(migraphx_module_t* out,
                                                            migraphx_program_t program)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out = object_cast<migraphx_module_t>((program->object).get_main_module());
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_create_module(migraphx_module_t* out, migraphx_program_t program, const char* name)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out = object_cast<migraphx_module_t>((program->object).create_module((name)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_compile(migraphx_program_t program,
                                                    migraphx_target_t target,
                                                    migraphx_compile_options_t options)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(target == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter target: Null pointer");
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        (program->object).compile((target->object), (options->object));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_get_parameter_shapes(migraphx_program_parameter_shapes_t* out,
                                      migraphx_program_t program)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out =
            allocate<migraphx_program_parameter_shapes_t>((program->object).get_parameter_shapes());
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_get_output_shapes(migraphx_shapes_t* out,
                                                              migraphx_program_t program)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out = allocate<migraphx_shapes_t>(migraphx::get_output_shapes((program->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_print(const_migraphx_program_t program)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        migraphx::print_program((program->object));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_sort(migraphx_program_t program)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        (program->object).sort();
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_program_run(migraphx_arguments_t* out,
                                                migraphx_program_t program,
                                                migraphx_program_parameters_t params)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(params == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter params: Null pointer");
        *out = allocate<migraphx_arguments_t>(migraphx::run((program->object), (params->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_equal(bool* out, const_migraphx_program_t program, const_migraphx_program_t x)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(x == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter x: Null pointer");
        *out = migraphx::equal((program->object), (x->object));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_program_experimental_get_context(migraphx_context_t* out, const_migraphx_program_t program)
{
    auto api_error_result = migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out = allocate<migraphx_context_t>(migraphx::get_context((program->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_operation_destroy(migraphx_operation_t operation)
{
    auto api_error_result = migraphx::try_([&] { destroy((operation)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_operation_assign_to(migraphx_operation_t output,
                                                        const_migraphx_operation_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_operation_create(migraphx_operation_t* operation,
                                                     const char* name,
                                                     const char* attributes,
                                                     ...)
{
    va_list vlist;
    va_start(vlist, attributes);
    auto api_error_result = migraphx::try_([&] {
        *operation = object_cast<migraphx_operation_t>(
            allocate<migraphx::operation>(migraphx::create_op((name), (attributes), (vlist))));
    });
    va_end(vlist);
    return api_error_result;
}

extern "C" migraphx_status
migraphx_operation_name(char* out, size_t out_size, migraphx_operation_t operation)
{
    auto api_error_result = migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(operation == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter operation: Null pointer");
        auto&& api_result = (operation->object).name();
        auto* it = std::copy_n(api_result.begin(), std::min(api_result.size(), out_size - 1), out);
        *it      = '\0';
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_load(migraphx_program_t* out, const char* name, migraphx_file_options_t options)
{
    auto api_error_result = migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(migraphx::load((name), (options->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_save(migraphx_program_t p, const char* name, migraphx_file_options_t options)
{
    auto api_error_result = migraphx::try_([&] {
        if(p == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter p: Null pointer");
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        migraphx::save((p->object), (name), (options->object));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_onnx_options_destroy(migraphx_onnx_options_t onnx_options)
{
    auto api_error_result = migraphx::try_([&] { destroy((onnx_options)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_onnx_options_assign_to(migraphx_onnx_options_t output,
                                                           const_migraphx_onnx_options_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_onnx_options_create(migraphx_onnx_options_t* onnx_options)
{
    auto api_error_result = migraphx::try_([&] {
        *onnx_options = object_cast<migraphx_onnx_options_t>(allocate<migraphx::onnx_options>());
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_onnx_options_set_input_parameter_shape(
    migraphx_onnx_options_t onnx_options, const char* name, size_t* dims, size_t dims_size)
{
    auto api_error_result = migraphx::try_([&] {
        if(onnx_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter onnx_options: Null pointer");
        if(dims == nullptr and dims_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter dims: Null pointer");
        migraphx::set_input_parameter_shape(
            (onnx_options->object), (name), (std::vector<size_t>(dims, dims + dims_size)));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_onnx_options_set_default_dim_value(migraphx_onnx_options_t onnx_options, size_t value)
{
    auto api_error_result = migraphx::try_([&] {
        if(onnx_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter onnx_options: Null pointer");
        migraphx::set_default_dim_value((onnx_options->object), (value));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_onnx_options_set_default_loop_iterations(migraphx_onnx_options_t onnx_options,
                                                  int64_t value)
{
    auto api_error_result = migraphx::try_([&] {
        if(onnx_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter onnx_options: Null pointer");
        migraphx::set_default_loop_iterations((onnx_options->object), (value));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_file_options_destroy(migraphx_file_options_t file_options)
{
    auto api_error_result = migraphx::try_([&] { destroy((file_options)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_file_options_assign_to(migraphx_file_options_t output,
                                                           const_migraphx_file_options_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_file_options_create(migraphx_file_options_t* file_options)
{
    auto api_error_result = migraphx::try_([&] {
        *file_options = object_cast<migraphx_file_options_t>(allocate<migraphx::file_options>());
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_file_options_set_file_format(migraphx_file_options_t file_options, const char* format)
{
    auto api_error_result = migraphx::try_([&] {
        if(file_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter file_options: Null pointer");
        migraphx::set_file_format((file_options->object), (format));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_compile_options_destroy(migraphx_compile_options_t compile_options)
{
    auto api_error_result = migraphx::try_([&] { destroy((compile_options)); });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_compile_options_assign_to(migraphx_compile_options_t output,
                                   const_migraphx_compile_options_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_compile_options_create(migraphx_compile_options_t* compile_options)
{
    auto api_error_result = migraphx::try_([&] {
        *compile_options =
            object_cast<migraphx_compile_options_t>(allocate<migraphx::compile_options>());
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_compile_options_set_offload_copy(migraphx_compile_options_t compile_options, bool value)
{
    auto api_error_result = migraphx::try_([&] {
        if(compile_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter compile_options: Null pointer");
        migraphx::set_offload_copy((compile_options->object), (value));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_compile_options_set_fast_math(migraphx_compile_options_t compile_options, bool value)
{
    auto api_error_result = migraphx::try_([&] {
        if(compile_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter compile_options: Null pointer");
        migraphx::set_fast_math((compile_options->object), (value));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_parse_onnx(migraphx_program_t* out, const char* name, migraphx_onnx_options_t options)
{
    auto api_error_result = migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(migraphx::parse_onnx((name), (options->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_parse_onnx_buffer(migraphx_program_t* out,
                                                      const void* data,
                                                      size_t size,
                                                      migraphx_onnx_options_t options)
{
    auto api_error_result = migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(
            migraphx::parse_onnx_buffer((data), (size), (options->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_tf_options_destroy(migraphx_tf_options_t tf_options)
{
    auto api_error_result = migraphx::try_([&] { destroy((tf_options)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_tf_options_assign_to(migraphx_tf_options_t output,
                                                         const_migraphx_tf_options_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_tf_options_create(migraphx_tf_options_t* tf_options)
{
    auto api_error_result = migraphx::try_([&] {
        *tf_options = object_cast<migraphx_tf_options_t>(allocate<migraphx::tf_options>());
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_tf_options_set_nhwc(migraphx_tf_options_t tf_options,
                                                        bool is_nhwc)
{
    auto api_error_result = migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        migraphx::set_nhwc((tf_options->object), (is_nhwc));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_tf_options_set_input_parameter_shape(
    migraphx_tf_options_t tf_options, const char* name, size_t* dims, size_t dims_size)
{
    auto api_error_result = migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        if(dims == nullptr and dims_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter dims: Null pointer");
        migraphx::set_input_parameter_shape(
            (tf_options->object), (name), (std::vector<size_t>(dims, dims + dims_size)));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_tf_options_set_default_dim_value(migraphx_tf_options_t tf_options, size_t value)
{
    auto api_error_result = migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        migraphx::set_default_dim_value((tf_options->object), (value));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_tf_options_set_output_names(migraphx_tf_options_t tf_options,
                                                                const char** names,
                                                                size_t names_size)
{
    auto api_error_result = migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        if(names == nullptr and names_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter names: Null pointer");
        migraphx::set_output_names((tf_options->object),
                                   (std::vector<const char*>(names, names + names_size)));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_parse_tf(migraphx_program_t* out, const char* name, migraphx_tf_options_t options)
{
    auto api_error_result = migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(migraphx::parse_tf((name), (options->object)));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_op_names_destroy(migraphx_quantize_op_names_t quantize_op_names)
{
    auto api_error_result = migraphx::try_([&] { destroy((quantize_op_names)); });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_op_names_assign_to(migraphx_quantize_op_names_t output,
                                     const_migraphx_quantize_op_names_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_op_names_create(migraphx_quantize_op_names_t* quantize_op_names)
{
    auto api_error_result = migraphx::try_([&] {
        *quantize_op_names =
            object_cast<migraphx_quantize_op_names_t>(allocate<std::vector<std::string>>());
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_op_names_add(migraphx_quantize_op_names_t quantize_op_names, const char* name)
{
    auto api_error_result = migraphx::try_([&] {
        if(quantize_op_names == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter quantize_op_names: Null pointer");
        (quantize_op_names->object).push_back((name));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_quantize_fp16_with_op_names(migraphx_program_t prog,
                                                                migraphx_quantize_op_names_t name)
{
    auto api_error_result = migraphx::try_([&] {
        if(prog == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter prog: Null pointer");
        if(name == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter name: Null pointer");
        migraphx::quantize_fp16_with_op_names((prog->object), (name->object));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_quantize_fp16(migraphx_program_t prog)
{
    auto api_error_result = migraphx::try_([&] {
        if(prog == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter prog: Null pointer");
        migraphx::quantize_fp16((prog->object));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_int8_options_destroy(migraphx_quantize_int8_options_t quantize_int8_options)
{
    auto api_error_result = migraphx::try_([&] { destroy((quantize_int8_options)); });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_int8_options_assign_to(migraphx_quantize_int8_options_t output,
                                         const_migraphx_quantize_int8_options_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_int8_options_create(migraphx_quantize_int8_options_t* quantize_int8_options)
{
    auto api_error_result = migraphx::try_([&] {
        *quantize_int8_options = object_cast<migraphx_quantize_int8_options_t>(
            allocate<migraphx::quantize_int8_options>());
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_quantize_int8_options_add_op_name(migraphx_quantize_int8_options_t quantize_int8_options,
                                           const char* name)
{
    auto api_error_result = migraphx::try_([&] {
        if(quantize_int8_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter quantize_int8_options: Null pointer");
        migraphx::add_op_name((quantize_int8_options->object), (name));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_quantize_int8_options_add_calibration_data(
    migraphx_quantize_int8_options_t quantize_int8_options, migraphx_program_parameters_t data)
{
    auto api_error_result = migraphx::try_([&] {
        if(quantize_int8_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter quantize_int8_options: Null pointer");
        if(data == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter data: Null pointer");
        migraphx::add_calibration_data((quantize_int8_options->object), (data->object));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_quantize_int8(migraphx_program_t prog,
                                                  migraphx_target_t target,
                                                  migraphx_quantize_int8_options_t options)
{
    auto api_error_result = migraphx::try_([&] {
        if(prog == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter prog: Null pointer");
        if(target == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter target: Null pointer");
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        migraphx::quantize_int8_wrap((prog->object), (target->object), (options->object));
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_context_destroy(migraphx_context_t context)
{
    auto api_error_result = migraphx::try_([&] { destroy((context)); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_context_assign_to(migraphx_context_t output,
                                                      const_migraphx_context_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status migraphx_context_create(migraphx_context_t* context)
{
    auto api_error_result = migraphx::try_(
        [&] { *context = object_cast<migraphx_context_t>(allocate<migraphx::context>()); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_context_finish(const_migraphx_context_t context)
{
    auto api_error_result = migraphx::try_([&] {
        if(context == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter context: Null pointer");
        (context->object).finish();
    });
    return api_error_result;
}

extern "C" migraphx_status migraphx_context_get_queue(void** out, migraphx_context_t context)
{
    auto api_error_result = migraphx::try_([&] {
        if(context == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter context: Null pointer");
        *out = (context->object).get_queue().unsafe_get();
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_experimental_custom_op_destroy(migraphx_experimental_custom_op_t experimental_custom_op)
{
    auto api_error_result = migraphx::try_([&] { destroy((experimental_custom_op)); });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_experimental_custom_op_assign_to(migraphx_experimental_custom_op_t output,
                                          const_migraphx_experimental_custom_op_t input)
{
    auto api_error_result = migraphx::try_([&] { *output = *input; });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_experimental_custom_op_create(migraphx_experimental_custom_op_t* experimental_custom_op,
                                       void* obj,
                                       migraphx_experimental_custom_op_copy c,
                                       migraphx_experimental_custom_op_delete d,
                                       const char* name)
{
    auto api_error_result = migraphx::try_([&] {
        *experimental_custom_op =
            allocate<migraphx_experimental_custom_op_t>((obj), (c), (d), (name));
    });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_experimental_custom_op_set_compute(migraphx_experimental_custom_op_t obj,
                                            migraphx_experimental_custom_op_compute input)
{
    auto api_error_result = migraphx::try_([&] { (obj)->compute_f = (input); });
    return api_error_result;
}

extern "C" migraphx_status migraphx_experimental_custom_op_set_compute_shape(
    migraphx_experimental_custom_op_t obj, migraphx_experimental_custom_op_compute_shape input)
{
    auto api_error_result = migraphx::try_([&] { (obj)->compute_shape_f = (input); });
    return api_error_result;
}

extern "C" migraphx_status
migraphx_experimental_custom_op_register(migraphx_experimental_custom_op_t experimental_custom_op)
{
    auto api_error_result = migraphx::try_([&] {
        if(experimental_custom_op == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter experimental_custom_op: Null pointer");
        migraphx::register_custom_op((*experimental_custom_op));
    });
    return api_error_result;
}
