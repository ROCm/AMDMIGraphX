#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/iterator_for.hpp>
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
    // A map is used to keep references to modules of the program
    std::map<std::string, module> modules;
    context ctx;
    std::string target_name;
};

static void print_instruction(std::ostream& os,
                              instruction_ref ins,
                              const std::unordered_map<instruction_ref, std::string>& names)
{
    os << names.at(ins) << " = ";

    os << ins->get_operator();

    if(ins->name() == "@literal")
    {
        if(ins->get_literal().get_shape().elements() > 10)
            os << "{ ... }";
        else
            os << "{" << ins->get_literal() << "}";
    }

    if(!ins->inputs().empty())
    {
        char delim = '(';
        for(auto&& arg : ins->inputs())
        {
            os << delim << names.at(arg);
            delim = ',';
        }
        os << ")";
    }

    // skip return instruction shape
    if(ins->name() != "@return")
        os << " -> " << ins->get_shape();
}

program::program() : impl(std::make_unique<program_impl>()) { impl->modules["main"] = {}; }

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
    const auto* mm = this->get_main_module();
    return mm->get_parameter_shape(std::move(name));
}

std::vector<std::string> program::get_parameter_names() const
{
    const auto* mm = this->get_main_module();
    return mm->get_parameter_names();
}

instruction_ref program::get_parameter(std::string name) const
{
    const auto* mm = this->get_main_module();
    return mm->get_parameter(std::move(name));
}

std::unordered_map<std::string, shape> program::get_parameter_shapes() const
{
    const auto* mm = this->get_main_module();
    return mm->get_parameter_shapes();
}

std::size_t program::size() const { return impl->modules.size(); }

std::vector<shape> program::get_output_shapes() const
{
    const auto* mm = this->get_main_module();
    return mm->get_output_shapes();
}

context& program::get_context() const { return impl->ctx; }

instruction_ref program::validate() const
{
    const auto* mm = this->get_main_module();
    return mm->validate();
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
    auto&& passes = t.get_passes(this->impl->ctx, options);

    for(auto& mp : impl->modules)
    {
        auto& modl = mp.second;
        assert(modl.validate() == modl.end());
        run_passes(modl, passes, options.trace);
        auto invalid = this->validate();
        if(invalid != modl.end())
        {
            auto index = std::distance(modl.begin(), invalid);
            MIGRAPHX_THROW("Invalid module " + mp.first + " from compilation at instruction " +
                           std::to_string(index));
        }
        modl.finalize(this->impl->ctx);
    }
}

void program::finalize()
{
    for(auto& mp : this->impl->modules)
    {
        mp.second.finalize(this->impl->ctx);
    }
}

template <class F>
std::vector<argument> generic_eval(const module& p,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   F trace)
{
    assert(p.validate() == p.end());
    std::unordered_map<instruction_ref, argument> results;
    results.reserve(p.size() * 2);
    std::vector<argument> values;
    values.reserve(16);
    for(auto ins : iterator_for(p))
    {
        const auto& name = ins->name();
        if(name == "@literal")
        {
            results.emplace(ins, trace(ins, [&] { return ins->get_literal().get_argument(); }));
        }
        else if(name == "@param")
        {
            results.emplace(
                ins, trace(ins, [&] {
                    auto param_name = any_cast<builtin::param>(ins->get_operator()).parameter;
                    if(not contains(params, param_name))
                        MIGRAPHX_THROW("Parameter not found: " + param_name);
                    auto param = params[param_name];
                    if(param.get_shape() != ins->get_shape())
                        MIGRAPHX_THROW("Incorrect shape {" + to_string(param.get_shape()) +
                                       "} for parameter: " + param_name);
                    return param;
                }));
        }
        else if(name == "@outline")
        {
            results.emplace(ins, trace(ins, [&] { return argument{ins->get_shape(), nullptr}; }));
        }
        else if(name == "@return")
        {
            std::vector<argument> prog_outputs;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(prog_outputs),
                           [&](instruction_ref i) {
                               assert(results.find(i) != results.end());
                               return results[i];
                           });

            return prog_outputs;
        }
        else
        {
            values.resize(ins->inputs().size());
            std::transform(
                ins->inputs().begin(), ins->inputs().end(), values.begin(), [&](instruction_ref i) {
                    assert(results.find(i) != results.end());
                    return results[i];
                });
            results.emplace(ins, trace(ins, [&] {
                                return ins->get_operator().compute(ctx, ins->get_shape(), values);
                            }));
        }
        assert(results.find(ins) != results.end());
    }

    return {results.at(std::prev(p.end()))};
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
    auto& ctx = this->impl->ctx;
#ifndef NDEBUG
    auto sctx          = ctx;
    auto check_context = [&](auto f) {
        assert(is_shared(ctx, sctx));
        auto x = f();
        sctx   = ctx;
        return x;
    };
#else
    auto check_context = [](auto f) { return f(); };
#endif

    auto trace_level = value_of(MIGRAPHX_TRACE_EVAL{});

    if(trace_level > 0)
    {
        return generic_eval(*this, ctx, std::move(params), [&](auto& ins, auto f) {
            ctx.finish();
            std::cout << "Run instruction: ";
            this->debug_print(ins);
            auto result = check_context(f);
            ctx.finish();
            if(trace_level > 1 and ins->name().front() != '@' and ins->name() != "load")
                std::cout << "Ouput: " << result << std::endl;
            return result;
        });
    }
    else
    {
        return generic_eval(
            *this, ctx, std::move(params), [&](auto&, auto f) { return check_context(f); });
    }
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
    {
        MIGRAPHX_THROW("Warning: Program version mismatch");
    }

    this->impl->target_name = v.at("target").to<std::string>();
    if(not this->impl->target_name.empty())
    {
        target t        = make_target(this->impl->target_name);
        this->impl->ctx = t.get_context();
        this->impl->ctx.from_value(v.at("context"));
    }

    auto val_modules = v.at("modules");
    for(const auto& vv : val_modules)
    {
        const auto& key = vv.get_key();
        auto val        = vv.without_key();
        module modl;
        modl.from_value(val);
        impl->modules[key] = modl;
    }
    this->finalize();
}

double common_average(const std::vector<double>& v)
{
    std::size_t n = v.size() / 4;
    double total  = std::accumulate(v.begin() + n, v.end() - n, 0.0);
    return total / std::distance(v.begin() + n, v.end() - n);
}

void program::perf_report(std::ostream& os, std::size_t n, parameter_map params) const
{
    using milliseconds = std::chrono::duration<double, std::milli>;
    auto& ctx          = this->impl->ctx;
    // Run once by itself
    eval(params);
    ctx.finish();
    // Run and time entire program
    std::vector<double> total_vec;
    total_vec.reserve(n);
    for(std::size_t i = 0; i < n; i++)
    {
        total_vec.push_back(time<milliseconds>([&] {
            eval(params);
            ctx.finish();
        }));
    }
    std::sort(total_vec.begin(), total_vec.end());
    std::unordered_map<instruction_ref, std::vector<double>> ins_vec;
    // Fill the map
    generic_eval(*this, ctx, params, [&](auto ins, auto) {
        ins_vec[ins].reserve(n);
        return argument{};
    });
    // Run and time each instruction
    for(std::size_t i = 0; i < n; i++)
    {
        generic_eval(*this, ctx, params, [&](auto ins, auto f) {
            argument result;
            ins_vec[ins].push_back(time<milliseconds>([&] {
                result = f();
                ctx.finish();
            }));
            return result;
        });
    }
    for(auto&& p : ins_vec)
        std::sort(p.second.begin(), p.second.end());
    // Run and time implicit overhead
    std::vector<double> overhead_vec;
    overhead_vec.reserve(n);
    for(std::size_t i = 0; i < n; i++)
    {
        overhead_vec.push_back(time<milliseconds>([&] { dry_run(params); }));
    }

    double total_time             = common_average(total_vec);
    double rate                   = 1000.0 / total_time;
    double overhead_time          = common_average(overhead_vec);
    double overhead_percent       = overhead_time * 100.0 / total_time;
    double total_instruction_time = 0.0;
    std::unordered_map<std::string, double> op_times;
    for(auto&& p : ins_vec)
    {
        double avg = common_average(p.second);
        op_times[p.first->name()] += avg;
        total_instruction_time += avg;
    }
    double calculate_overhead_time    = total_time - total_instruction_time;
    double calculate_overhead_percent = calculate_overhead_time * 100.0 / total_time;

    this->debug_print([&](auto ins, auto names) {
        print_instruction(std::cout, ins, names);

        // skip return instruction
        if(ins->name() == "@return")
            return;

        double avg     = common_average(ins_vec[ins]);
        double percent = std::ceil(100.0 * avg / total_instruction_time);
        os << ": " << avg << "ms, " << percent << "%";
        os << std::endl;
    });

    os << std::endl;
    os << "Summary:" << std::endl;
    std::vector<std::pair<double, std::string>> op_times_sorted;
    std::transform(op_times.begin(),
                   op_times.end(),
                   std::back_inserter(op_times_sorted),
                   [](auto p) { return std::make_pair(p.second, p.first); });
    std::sort(op_times_sorted.begin(), op_times_sorted.end(), std::greater<>{});
    for(auto&& p : op_times_sorted)
    {
        auto&& name    = p.second;
        double avg     = p.first;
        double percent = std::ceil(100.0 * avg / total_instruction_time);
        os << name << ": " << avg << "ms, " << percent << "%" << std::endl;
    }

    os << std::endl;

    os << "Rate: " << rate << "/sec" << std::endl;
    os << "Total time: " << total_time << "ms" << std::endl;
    os << "Total instructions time: " << total_instruction_time << "ms" << std::endl;
    os << "Overhead time: " << overhead_time << "ms"
       << ", " << calculate_overhead_time << "ms" << std::endl;
    os << "Overhead: " << std::round(overhead_percent) << "%"
       << ", " << std::round(calculate_overhead_percent) << "%" << std::endl;
}

void program::debug_print() const { std::cout << *this << std::endl; }
void program::debug_print(instruction_ref ins) const
{
    if(std::any_of(this->impl->modules.begin(), this->impl->modules.end(), [&](auto it) {
           return (it.second.end() == ins);
       }))
    {
        std::cout << "End instruction" << std::endl;
        return;
    }
    else if(not std::any_of(this->impl->modules.begin(), this->impl->modules.end(), [&](auto it) {
                return it.second.has_instruction(ins);
            }))
    {
        std::cout << "Instruction not part of program" << std::endl;
        return;
    }

    std::stringstream ss;
    this->debug_print([&](auto x, const auto& names) {
        if(x == ins)
        {
            print_instruction(std::cout, x, names);
            std::cout << std::endl;
        }
    });
}

void program::debug_print(
    const std::function<void(instruction_ref,
                             const std::unordered_map<instruction_ref, std::string>&)>& print_func)
    const
{
    for(const auto& mdl : this->impl->modules)
    {
        mdl.second.debug_print(print_func);
    }
}

void program::print_graph(std::ostream& os, bool brief) const
{
    const auto* mm = this->get_main_module();
    mm->print_graph(os, brief);
}

void program::print_cpp(std::ostream& os) const
{
    os << "migraphx::program p;" << std::endl;
    const auto* mm = this->get_main_module();
    mm->print_cpp(os);
}

void program::dry_run(std::unordered_map<std::string, argument> params) const
{
    auto& ctx = this->impl->ctx;
    generic_eval(*this, ctx, std::move(params), [](auto&&...) { return argument{}; });
}

void program::annotate(std::ostream& os, const std::function<void(instruction_ref)>& a) const
{
    for(auto& modl : this->impl->modules)
    {
        std::cout << modl.first << ":" << std::endl;
        modl.second.annotate(os, a);
    }
}

module* program::get_main_module() { return &impl->modules["main"]; }

const module* program::get_main_module() const { return &impl->modules["main"]; }

program& program::sort()
{
    for(auto& modl : this->impl->modules)
    {
        modl.second.sort();
    }

    return *this;
}

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    for(auto& mp : p.impl->modules)
    {
        os << "Module " << mp.first << ": " << std::endl;
        os << mp.second;
        os << std::endl;
    }

    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
