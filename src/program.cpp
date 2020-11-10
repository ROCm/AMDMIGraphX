#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <set>
#include <utility>
#include <unordered_set>
#include <map>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program_impl
{
    // A list is used to keep references to modules of the program
    std::map<std::string, module> modules;
    context ctx;
    std::string target_name;
};

program::program() { impl->modules["main"] = {}; }

program::program(program&&) noexcept = default;
program::~program() noexcept         = default;

// copy constructor
program::program(const program& p) { assign(p); }

// copy assignment operator
program& program::operator=(program p)
{
    std::swap(p.impl, this->impl);
    return *this;
}

void program::assign(const program& p)
{
    if(!impl)
    {
        impl = std::make_unique<program_impl>();
    }
    else if(!impl->modules.empty())
    {
        impl->modules.clear();
    }
    impl->ctx         = p.impl->ctx;
    impl->target_name = p.impl->target_name;

    for(auto& modl_pair : p.impl->modules)
    {
        impl->modules[modl_pair.first] = module(modl_pair.second);
    }
}

shape program::get_parameter_shape(std::string name) const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].get_parameter_shape(std::move(name));
}

std::vector<std::string> program::get_parameter_names() const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].get_parameter_names();
}

instruction_ref program::get_parameter(std::string name) const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].get_parameter(std::move(name));
}

std::unordered_map<std::string, shape> program::get_parameter_shapes() const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].get_parameter_shapes();
}

bool program::has_instruction(instruction_ref ins) const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].has_instruction(ins);
}

std::size_t program::size() const { return impl->modules.size(); }

instruction_ref program::begin() const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].begin();
}

instruction_ref program::end() const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].end();
}

std::vector<shape> program::get_output_shapes() const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].get_output_shapes();
}

context& program::get_context() const { return impl->ctx; }

instruction_ref program::validate() const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].validate();
}

bool program::is_compiled() const { return not this->impl->target_name.empty(); }

void program::compile(const target& t, compile_options options)
{
    assert(not this->is_compiled());
    this->impl->target_name = t.name();
    this->impl->ctx         = t.get_context();
    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
        options.trace = tracer{std::cout};

    options.trace(*this);
    options.trace();

    for(auto& mp : impl->modules)
    {
        auto& m = mp.second;
        m.compile(t, options);
    }
}

void program::finalize()
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].finalize();
}

template <class F>
std::vector<argument> generic_eval(const program& p,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   F trace)
{
    const auto* mm = p.get_main_module();
    return generic_eval(*mm, ctx, params, trace);
}

std::vector<argument> program::eval(parameter_map params) const
{
    assert(contains(impl->modules, "main"));
    return impl->modules["main"].eval(std::move(params));
}

const int program_file_version = 2;

value program::to_value() const
{
    value result;
    result["version"] = program_file_version;
    result["target"]  = this->impl->target_name;
    if(not this->impl->target_name.empty())
        result["context"] = this->impl->ctx.to_value();

    result["modules"] = value::object{};
    auto& module_val  = result.at("modules");
    for(auto& m : impl->modules)
    {
        module_val[m.first] = m.second.to_value();
    }
    return result;
}

void program::from_value(const value& v)
{
    auto version = v.at("version").to<int>();
    if(version != program_file_version)
        std::cout << "Warning: Program version mismatch" << std::endl;

    this->impl->target_name = v.at("target").to<std::string>();
    if(not this->impl->target_name.empty())
    {
        target t        = make_target(this->impl->target_name);
        this->impl->ctx = t.get_context();
        this->impl->ctx.from_value(v.at("context"));
    }

    auto val_modules = v.at("modules");
    for(auto vv : val_modules)
    {
        auto key = vv.get_key();
        auto val = vv.without_key();
        module modl;
        modl.from_value(val);
        impl->modules[key] = modl;
    }
}

void program::perf_report(std::ostream& os, std::size_t n, parameter_map params) const
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].perf_report(os, n, std::move(params));
}

void program::debug_print() const { std::cout << *this << std::endl; }

void program::debug_print(instruction_ref ins) const
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].debug_print(ins);
}

void program::debug_print(const std::vector<instruction_ref>& inss) const
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].debug_print(inss);
}

void program::print_graph(std::ostream& os, bool brief) const
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].print_graph(os, brief);
}

void program::print_cpp(std::ostream& os) const
{
    os << "migraphx::program p;" << std::endl;
    assert(contains(impl->modules, "main"));
    impl->modules["main"].print_cpp(os);
}

void program::dry_run(std::unordered_map<std::string, argument> params) const
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].dry_run(std::move(params));
}

void program::annotate(std::ostream& os, std::function<void(instruction_ref)> a) const
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].annotate(os, std::move(a));
}

module* program::get_main_module()
{
    if(!contains(impl->modules, "main"))
    {
        impl->modules["main"] = {};
    }

    return &impl->modules["main"];
}

const module* program::get_main_module() const
{
    assert(contains(impl->modules, "main"));
    return &impl->modules["main"];
}

program& program::sort()
{
    assert(contains(impl->modules, "main"));
    impl->modules["main"].sort();
    return *this;
}

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    for(auto& mp : p.impl->modules)
    {
        std::cout << "Module " << mp.first << ": " << std::endl;
        os << mp.second;
        os << std::endl;
    }

    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
