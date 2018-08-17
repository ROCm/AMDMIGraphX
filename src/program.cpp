#include <migraph/program.hpp>
#include <migraph/stringutils.hpp>
#include <migraph/instruction.hpp>
#include <migraph/env.hpp>
#include <migraph/time.hpp>
#include <migraph/iterator_for.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace migraph {

MIGRAPH_DECLARE_ENV_VAR(MIGRAPH_TRACE_COMPILE)

struct program_impl
{
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;
    context ctx;
};

const operation& get_operation(instruction_ref ins) { return ins->op; }

template <class F>
static void print_program(std::ostream& os, const program& p, F annonate)
{
    std::unordered_map<instruction_ref, std::string> names;
    int count = 0;

    for(auto ins : iterator_for(p))
    {
        std::string var_name = "@" + std::to_string(count);
        if(ins->op.name() == "@param")
        {
            var_name = any_cast<builtin::param>(ins->op).parameter;
        }

        os << var_name << " = ";

        os << ins->op;

        if(ins->op.name() == "@literal")
        {
            if(ins->lit.get_shape().elements() > 10)
                os << "{ ... }";
            else
                os << "{" << ins->lit << "}";
        }

        if(!ins->arguments.empty())
        {
            char delim = '(';
            for(auto&& arg : ins->arguments)
            {
                assert(p.has_instruction(arg) && "Instruction not found");
                os << delim << names.at(arg);
                delim = ',';
            }
            os << ")";
        }

        os << " -> " << ins->result;

        annonate(ins, names);

        os << std::endl;

        names.emplace(ins, var_name);
        count++;
    }
}

program::program() : impl(std::make_unique<program_impl>()) {}

program::program(program&&) noexcept = default;
program& program::operator=(program&&) noexcept = default;
program::~program() noexcept                    = default;

instruction_ref program::add_instruction(operation op, std::vector<instruction_ref> args)
{
    return insert_instruction(impl->instructions.end(), std::move(op), std::move(args));
}
instruction_ref
program::insert_instruction(instruction_ref ins, operation op, std::vector<instruction_ref> args)
{
    assert(std::all_of(
               args.begin(), args.end(), [&](instruction_ref x) { return has_instruction(x); }) &&
           "Argument is not an exisiting instruction");
    assert(not starts_with(op.name(), "@"));
    // TODO: Use move
    shape r     = compute_shape(op, args);
    auto result = impl->instructions.insert(ins, {op, r, args});
    backreference(result);
    assert(result->arguments == args);
    assert(result->valid(begin()));
    return result;
}

instruction_ref
program::replace_instruction(instruction_ref ins, operation op, std::vector<instruction_ref> args)
{
    assert(std::all_of(
               args.begin(), args.end(), [&](instruction_ref x) { return has_instruction(x); }) &&
           "Argument is not an exisiting instruction");
    assert(not starts_with(op.name(), "@"));

    shape r = compute_shape(op, args);
    ins->replace(op, r, args);
    backreference(ins);
    assert(ins->valid(begin()));
    return ins;
}

instruction_ref program::replace_instruction(instruction_ref ins, instruction_ref rep)
{
    assert(has_instruction(ins));
    assert(has_instruction(rep));
    assert(ins != rep);
    // TODO: Should it be an error if the output is empty?
    if(ins->output.empty())
    {
        return rep;
    }
    for(auto&& out : ins->output)
    {
        // TODO: Check for possible cycles
        if(out != rep)
        {
            replace_argument(out, ins, rep);
        }
        assert(out->valid(begin()));
    }
    // Replacement should not be dead code unless its the last instruction
    assert(!rep->output.empty() or rep == std::prev(end()));
    assert(ins->valid(begin()));
    assert(rep->valid(begin()));
    return rep;
}

instruction_ref program::remove_instruction(instruction_ref ins)
{
    assert(has_instruction(ins));
    assert(ins->output.empty());
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
    assert(std::all_of(first, last, [&](instruction& ins) { return ins.output.empty(); }));
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

instruction_ref program::add_outline(shape s)
{
    impl->instructions.push_front({builtin::outline{s}, s, {}});
    return impl->instructions.begin();
}

instruction_ref program::add_parameter(std::string name, shape s)
{
    impl->instructions.push_front({builtin::param{std::move(name)}, s, {}});
    return impl->instructions.begin();
}

shape program::get_parameter_shape(std::string name) const
{
    auto ins = std::find_if(
        impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
            if(x.op.name() == "@param")
            {
                return any_cast<builtin::param>(x.op).parameter == name;
            }
            else
            {
                return false;
            }
        });
    if(ins != this->end())
        return ins->result;
    else
        return {};
}

std::unordered_map<std::string, shape> program::get_parameter_shapes() const
{
    std::unordered_map<std::string, shape> result;
    for(auto&& ins : impl->instructions)
    {
        if(ins.op.name() == "@param")
        {
            auto&& name  = any_cast<builtin::param>(ins.op).parameter;
            result[name] = ins.result;
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

shape program::get_shape() const { return impl->instructions.back().result; }

instruction_ref program::validate() const
{
    return std::find_if(impl->instructions.begin(),
                        impl->instructions.end(),
                        [&](const instruction& i) { return !i.valid(impl->instructions.begin()); });
}

void program::compile(const target& t)
{
    assert(this->validate() == impl->instructions.end());
    this->impl->ctx = t.get_context();
    if(enabled(MIGRAPH_TRACE_COMPILE{}))
        std::cout << *this << std::endl << std::endl;
    ;
    for(auto&& p : t.get_passes(this->impl->ctx))
    {
        if(enabled(MIGRAPH_TRACE_COMPILE{}))
            std::cout << "Pass: " << p.name() << std::endl;
        p.apply(*this);
        if(enabled(MIGRAPH_TRACE_COMPILE{}))
            std::cout << *this << std::endl;
#ifndef NDEBUG
        if(enabled(MIGRAPH_TRACE_COMPILE{}))
            std::cout << "Validate ..." << std::endl;
        auto invalid = this->validate();
        if(invalid != impl->instructions.end())
        {
            auto index = std::distance(impl->instructions.begin(), invalid);
            MIGRAPH_THROW(p.name() + " pass produces invalid program at instruction " +
                          std::to_string(index) + ": " + invalid->op.name());
        }
        if(enabled(MIGRAPH_TRACE_COMPILE{}))
            std::cout << std::endl;
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
        if(ins->op.name() == "@literal")
        {
            results.emplace(ins, trace(ins, [&] { return ins->lit.get_argument(); }));
        }
        else if(ins->op.name() == "@param")
        {
            results.emplace(ins, trace(ins, [&] {
                                return params.at(any_cast<builtin::param>(ins->op).parameter);
                            }));
        }
        else if(ins->op.name() == "@outline")
        {
            results.emplace(ins, trace(ins, [&] { return argument{ins->result, nullptr}; }));
        }
        else
        {
            values.resize(ins->arguments.size());
            std::transform(ins->arguments.begin(),
                           ins->arguments.end(),
                           values.begin(),
                           [&](instruction_ref i) {
                               assert(results.find(i) != results.end());
                               return results[i];
                           });
            results.emplace(ins,
                            trace(ins, [&] { return ins->op.compute(ctx, ins->result, values); }));
        }
        assert(results.find(ins) != results.end());
    }
    return results.at(std::prev(p.end()));
}

argument program::eval(std::unordered_map<std::string, argument> params) const
{
    return generic_eval(*this, this->impl->ctx, params, [](auto&, auto f) { return f(); });
}

double common_average(const std::vector<double>& v)
{
    std::size_t n = v.size() / 4;
    double total = std::accumulate(v.begin()+n, v.end()-n, 0.0);
    return total / std::distance(v.begin()+n, v.end()-n);
}

void program::perf_report(std::ostream& os, std::size_t n, parameter_map params) const
{
    using milliseconds = std::chrono::duration<double, std::milli>;
    // Run once by itself
    eval(params);
    // Run and time entire program
    std::vector<double> total_vec;
    total_vec.reserve(n);
    for(std::size_t i = 0; i < n; i++)
    {
        total_vec.push_back(time<milliseconds>([&] { eval(params); }));
    }
    std::sort(total_vec.begin(), total_vec.end());
    std::unordered_map<instruction_ref, std::vector<double>> ins_vec;
    // Fill the map
    generic_eval(*this, this->impl->ctx, params, [&](auto ins, auto) {
        ins_vec[ins].reserve(n);
        return argument{};
    });
    // Run and time each instruction
    for(std::size_t i = 0; i < n; i++)
    {
        generic_eval(*this, this->impl->ctx, params, [&](auto ins, auto f) {
            argument result;
            ins_vec[ins].push_back(time<milliseconds>([&] { result = f(); }));
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
        overhead_vec.push_back(time<milliseconds>([&] {
            generic_eval(*this, this->impl->ctx, params, [](auto...) { return argument{}; });
        }));
    }

    double total_time             = common_average(total_vec);
    double rate = std::ceil(1000.0 / total_time);
    double overhead_time          = common_average(overhead_vec);
    double overhead_percent       = overhead_time * 100.0 / total_time;
    double total_instruction_time = 0.0;
    for(auto&& p : ins_vec)
        total_instruction_time += common_average(p.second);
    double calculate_overhead_time    = total_time - total_instruction_time;
    double calculate_overhead_percent = calculate_overhead_time * 100.0 / total_time;

    print_program(os, *this, [&](auto ins, auto&&) { os << ": " << common_average(ins_vec[ins]) << "ms"; });

    os << "Rate: " << rate << "/sec" << std::endl;
    os << "Total time: " << total_time << "ms" << std::endl;
    os << "Total instructions time: " << total_instruction_time << "ms" << std::endl;
    os << "Overhead time: " << overhead_time << "ms"
       << ", " << calculate_overhead_time << "ms" << std::endl;
    os << "Overhead: " << std::round(overhead_percent) << "%"
       << ", " << std::round(calculate_overhead_percent) << "%" << std::endl;
}

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    print_program(os, p, [](auto&&...) {});
    return os;
}

} // namespace migraph
