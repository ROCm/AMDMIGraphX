#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

MIGRAPH_DECLARE_ENV_VAR(MIGRAPH_TRACE_COMPILE)
MIGRAPH_DECLARE_ENV_VAR(MIGRAPH_TRACE_EVAL)

struct program_impl
{
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;
    context ctx;
};

const operation& get_operation(instruction_ref ins) { return ins->get_operator(); }

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

    os << " -> " << ins->get_shape();
}

template <class F>
static void print_program(std::ostream& os, const program& p, F annonate)
{
    std::unordered_map<instruction_ref, std::string> names;
    int count = 0;

    for(auto ins : iterator_for(p))
    {
        std::string var_name = "@" + std::to_string(count);
        if(ins->name() == "@param")
        {
            var_name = any_cast<builtin::param>(ins->get_operator()).parameter;
        }
        names.emplace(ins, var_name);

        // TODO: Use all_of
        for(auto&& arg : ins->inputs())
        {
            assert(p.has_instruction(arg) && "Instruction not found");
            (void)arg;
        }

        print_instruction(os, ins, names);

        annonate(ins, names);

        os << std::endl;

        count++;
    }
}

program::program() : impl(std::make_unique<program_impl>()) {}

program::program(program&&) noexcept = default;
program& program::operator=(program&&) noexcept = default;
program::~program() noexcept                    = default;

instruction_ref program::add_instruction(const operation& op, std::vector<instruction_ref> args)
{
    return insert_instruction(impl->instructions.end(), op, std::move(args));
}
instruction_ref program::insert_instruction(instruction_ref ins,
                                            const operation& op,
                                            std::vector<instruction_ref> args)
{
    assert(std::all_of(
               args.begin(), args.end(), [&](instruction_ref x) { return has_instruction(x); }) &&
           "Argument is not an exisiting instruction");
    assert(not starts_with(op.name(), "@"));
    // TODO: Use move
    shape r     = compute_shape(op, args);
    auto result = impl->instructions.insert(ins, {op, r, std::move(args)});
    instruction::backreference(result);
    // assert(result->inputs() == args);
    assert(result->valid(begin()));
    return result;
}

instruction_ref program::replace_instruction(instruction_ref ins,
                                             const operation& op,
                                             std::vector<instruction_ref> args)
{
    assert(std::all_of(
               args.begin(), args.end(), [&](instruction_ref x) { return has_instruction(x); }) &&
           "Argument is not an exisiting instruction");
    assert(not starts_with(op.name(), "@"));

    shape r = compute_shape(op, args);
    instruction::replace(ins, op, r, std::move(args));
    assert(ins->valid(begin()));
    return ins;
}

instruction_ref program::replace_instruction(instruction_ref ins, instruction_ref rep)
{
    assert(has_instruction(ins));
    assert(has_instruction(rep));
    assert(ins != rep);
    // TODO: Should it be an error if the output is empty?
    if(ins->outputs().empty())
    {
        return rep;
    }
    // Make a copy of outputs which can be changed when calling replace_argument
    auto outputs = ins->outputs();
    for(auto out : outputs)
    {
        // TODO: Check for possible cycles
        if(out != rep)
        {
            instruction::replace_argument(out, ins, rep);
        }
        assert(out->valid(begin()));
    }
    // Replacement should not be dead code unless its the last instruction
    assert(!rep->outputs().empty() or rep == std::prev(end()));
    // Output of the original instruction should only be the replacement or empty
    assert(ins->outputs().empty() or std::all_of(ins->outputs().begin(),
                                                 ins->outputs().end(),
                                                 [&](auto i) { return i == rep; }));
    assert(ins->valid(begin()));
    assert(rep->valid(begin()));
    return rep;
}

instruction_ref program::remove_instruction(instruction_ref ins)
{
    assert(has_instruction(ins));
    assert(ins->outputs().empty());
    ins->clear_arguments();
    return impl->instructions.erase(ins);
}

instruction_ref program::remove_instructions(instruction_ref first, instruction_ref last)
{
    if(first == last)
        return first;
    // TODO: Check every element
    assert(has_instruction(first));
    std::for_each(first, last, [&](instruction& ins) { ins.clear_arguments(); });
    assert(std::all_of(first, last, [&](instruction& ins) { return ins.outputs().empty(); }));
    return impl->instructions.erase(first, last);
}

instruction_ref program::move_instruction(instruction_ref src, instruction_ref dst)
{
    impl->instructions.splice(dst, impl->instructions, src);
    return src;
}

instruction_ref program::add_literal(literal l)
{
    impl->instructions.emplace_front(std::move(l));
    return impl->instructions.begin();
}

instruction_ref program::add_outline(const shape& s)
{
    impl->instructions.push_front({builtin::outline{s}, s, {}});
    return impl->instructions.begin();
}

instruction_ref program::add_parameter(std::string name, shape s)
{
    assert(get_parameter_shape(name) == shape{});
    impl->instructions.push_front({builtin::param{std::move(name)}, std::move(s), {}});
    return impl->instructions.begin();
}

shape program::get_parameter_shape(std::string name) const
{
    auto ins = std::find_if(
        impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
            if(x.name() == "@param")
            {
                return any_cast<builtin::param>(x.get_operator()).parameter == name;
            }
            else
            {
                return false;
            }
        });
    if(ins != this->end())
        return ins->get_shape();
    else
        return {};
}

instruction_ref program::get_parameter(std::string name) const
{
    auto ins = std::find_if(
        impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
            if(x.name() == "@param")
            {
                return any_cast<builtin::param>(x.get_operator()).parameter == name;
            }
            else
            {
                return false;
            }
        });
    if(ins != this->end())
        return ins;
    else
        return this->end();
}

std::unordered_map<std::string, shape> program::get_parameter_shapes() const
{
    std::unordered_map<std::string, shape> result;
    for(auto&& ins : impl->instructions)
    {
        if(ins.name() == "@param")
        {
            auto&& name  = any_cast<builtin::param>(ins.get_operator()).parameter;
            result[name] = ins.get_shape();
        }
    }
    return result;
}

bool program::has_instruction(instruction_ref ins) const
{
    return std::find_if(
               impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
                   return std::addressof(*ins) == std::addressof(x);
               }) != impl->instructions.end();
}

std::size_t program::size() const { return impl->instructions.size(); }
instruction_ref program::begin() const { return impl->instructions.begin(); }
instruction_ref program::end() const { return impl->instructions.end(); }

shape program::get_shape() const { return impl->instructions.back().get_shape(); }

instruction_ref program::validate() const
{
    return std::find_if(impl->instructions.begin(),
                        impl->instructions.end(),
                        [&](const instruction& i) { return !i.valid(impl->instructions.begin()); });
}

void program::compile(const target& t, tracer trace)
{
    assert(this->validate() == impl->instructions.end());
    this->impl->ctx = t.get_context();
    if(enabled(MIGRAPH_TRACE_COMPILE{}))
        trace = tracer{std::cout};
    trace(*this);
    trace();
    for(auto&& p : t.get_passes(this->impl->ctx))
    {
        trace("Pass: ", p.name());
        p.apply(*this);
        trace(*this);
#ifndef NDEBUG
        trace("Validate ...");
        auto invalid = this->validate();
        if(invalid != impl->instructions.end())
        {
            auto index = std::distance(impl->instructions.begin(), invalid);
            MIGRAPH_THROW(p.name() + " pass produces invalid program at instruction " +
                          std::to_string(index) + ": " + invalid->name());
        }
        trace();
#endif
    }
    auto invalid = this->validate();
    if(invalid != impl->instructions.end())
    {
        auto index = std::distance(impl->instructions.begin(), invalid);
        MIGRAPH_THROW("Invalid program from compilation at instruction " + std::to_string(index));
    }
}

template <class F>
argument generic_eval(const program& p,
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
        if(ins->name() == "@literal")
        {
            results.emplace(ins, trace(ins, [&] { return ins->get_literal().get_argument(); }));
        }
        else if(ins->name() == "@param")
        {
            results.emplace(ins, trace(ins, [&] {
                                auto param_name =
                                    any_cast<builtin::param>(ins->get_operator()).parameter;
                                if(not contains(params, param_name))
                                    MIGRAPH_THROW("Parameter not found: " + param_name);
                                return params.at(param_name);
                            }));
        }
        else if(ins->name() == "@outline")
        {
            results.emplace(ins, trace(ins, [&] { return argument{ins->get_shape(), nullptr}; }));
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
    return results.at(std::prev(p.end()));
}

argument program::eval(std::unordered_map<std::string, argument> params) const
{
    if(enabled(MIGRAPH_TRACE_EVAL{}))
    {
        auto& ctx = this->impl->ctx;
        return generic_eval(*this, this->impl->ctx, std::move(params), [&](auto& ins, auto f) {
            ctx.finish();
            std::cout << "Run instruction: ";
            this->debug_print(ins);
            return f();
        });
    }
    else
    {
        return generic_eval(
            *this, this->impl->ctx, std::move(params), [](auto&, auto f) { return f(); });
    }
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
        overhead_vec.push_back(time<milliseconds>(
            [&] { generic_eval(*this, ctx, params, [](auto...) { return argument{}; }); }));
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

    print_program(os, *this, [&](auto ins, auto&&) {
        double avg     = common_average(ins_vec[ins]);
        double percent = std::ceil(100.0 * avg / total_instruction_time);
        os << ": " << avg << "ms, " << percent << "%";
    });

    os << std::endl;
    os << "Summary:" << std::endl;
    for(auto&& p : op_times)
    {
        auto&& name    = p.first;
        double avg     = p.second;
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
    std::stringstream ss;
    print_program(ss, *this, [&](auto x, auto&& names) {
        if(x == ins)
        {
            print_instruction(std::cout, x, names);
            std::cout << std::endl;
        }
    });
}
void program::debug_print(const std::vector<instruction_ref>& inss) const
{
    for(auto ins : inss)
        debug_print(ins);
    std::cout << std::endl;
}

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    print_program(os, p, [](auto&&...) {});
    return os;
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
