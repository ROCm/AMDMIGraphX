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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// template <class F>
// static void print_program(const program& p, F print_func)
// {
//     std::unordered_map<instruction_ref, std::string> names;
//     int count = 0;

//     auto* mm = p.get_main_module();
//     for(auto ins : iterator_for(*mm))
//     {
//         std::string var_name;
//         if(ins->name() == "@param")
//         {
//             var_name = any_cast<builtin::param>(ins->get_operator()).parameter;
//         }
//         else
//         {
//             var_name = "@" + std::to_string(count);
//             count++;
//         }
//         names.emplace(ins, var_name);

//         // TODO: Use all_of
//         for(auto&& arg : ins->inputs())
//         {
//             assert(p.has_instruction(arg) && "Instruction not found");
//             (void)arg;
//         }

//         print_func(ins, names);
//     }
// }

program::program() {}

program::program(program&&) noexcept = default;
program::~program() noexcept         = default;

// copy constructor
program::program(const program& p) { assign(p); }

// copy assignment operator
program& program::operator=(program p)
{
    std::swap(p.main_module, this->main_module);
    return *this;
}

void program::assign(const program& p) { main_module = p.main_module; }

shape program::get_parameter_shape(std::string name) const
{
    return main_module.get_parameter_shape(name);
}

std::vector<std::string> program::get_parameter_names() const
{
    return main_module.get_parameter_names();
}

instruction_ref program::get_parameter(std::string name) const
{
    return main_module.get_parameter(name);
}

std::unordered_map<std::string, shape> program::get_parameter_shapes() const
{
    return main_module.get_parameter_shapes();
}

bool program::has_instruction(instruction_ref ins) const
{
    return main_module.has_instruction(ins);
}

std::size_t program::size() const { return main_module.size(); }
instruction_ref program::begin() const { return main_module.begin(); }
instruction_ref program::end() const { return main_module.end(); }

std::vector<shape> program::get_output_shapes() const { return main_module.get_output_shapes(); }

context& program::get_context() const { return main_module.get_context(); }

instruction_ref program::validate() const { return main_module.validate(); }

bool program::is_compiled() const { return main_module.is_compiled(); }

void program::compile(const target& t, compile_options options) { main_module.compile(t, options); }

void program::finalize() { main_module.finalize(); }

template <class F>
std::vector<argument> generic_eval(const program& p,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   F trace)
{
    const auto* mm = p.get_main_module();
    return generic_eval(*mm, ctx, params, trace);
}

std::vector<argument> program::eval(parameter_map params) const { return main_module.eval(params); }

const int program_file_version = 1;

value program::to_value() const {
    value result;
    result["version"] = program_file_version;
    result["main_module"] = main_module.to_value();
    return result;
}

void program::from_value(const value& v) {
    auto version = v.at("version").to<int>();
    if(version != program_file_version)
        std::cout << "Warning: Program version mismatch" << std::endl;
    
    auto mm_val = v.at("main_module").without_key();
    main_module.from_value(mm_val); 
}

void program::perf_report(std::ostream& os, std::size_t n, parameter_map params) const
{
    main_module.perf_report(os, n, params);
}

void program::debug_print() const { main_module.debug_print(); }
void program::debug_print(instruction_ref ins) const { main_module.debug_print(ins); }
void program::debug_print(const std::vector<instruction_ref>& inss) const
{
    main_module.debug_print(inss);
}

// static std::string enclose_name(const std::string& name)
// {
//     return '"' + replace_string(name, "\"", "\\\"") + '"';
// }

void program::print_graph(std::ostream& os, bool brief) const
{
    main_module.print_graph(os, brief);
}

// static std::string cpp_var_name(const std::string& name)
// {
//     return "m" + replace_string(name, "@", "x");
// }

// static std::string cpp_op_var(const std::string& name, instruction_ref ins)
// {
//     return replace_string(name, "@", ins->name());
// }

void program::print_cpp(std::ostream& os) const
{
    os << "migraphx::program p;" << std::endl;
    main_module.print_cpp(os);
}

void program::dry_run(std::unordered_map<std::string, argument> params) const
{
    main_module.dry_run(params);
}

void program::annotate(std::ostream& os, std::function<void(instruction_ref)> a) const
{
    main_module.annotate(os, a);
}

program& program::sort()
{
    main_module.sort();
    return *this;
}

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    const auto *mm = p.get_main_module();
    os << *mm;
    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
