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
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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
static void print_program(const program& p, F print_func)
{
    std::unordered_map<instruction_ref, std::string> names;
    int count = 0;

    for(auto ins : iterator_for(p))
    {
        std::string var_name;
        if(ins->name() == "@param")
        {
            var_name = any_cast<builtin::param>(ins->get_operator()).parameter;
        }
        else
        {
            var_name = "@" + std::to_string(count);
            count++;
        }
        names.emplace(ins, var_name);

        // TODO: Use all_of
        for(auto&& arg : ins->inputs())
        {
            assert(p.has_instruction(arg) && "Instruction not found");
            (void)arg;
        }

        print_func(ins, names);
    }
}

program::program() : impl(std::make_unique<program_impl>()) {}

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
    // clean the current program
    if(!impl)
    {
        impl = std::make_unique<program_impl>();
    }
    else if(!impl->instructions.empty())
    {
        impl->instructions.clear();
    }
    impl->ctx = p.impl->ctx;

    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto ins : iterator_for(p))
    {
        instruction_ref copy_ins{};
        if(ins->name() == "@literal")
        {
            auto l   = ins->get_literal();
            copy_ins = impl->instructions.insert(impl->instructions.end(), instruction{l});
        }
        else if(ins->name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(ins->get_operator()).parameter;
            auto s      = ins->get_shape();
            copy_ins    = impl->instructions.insert(impl->instructions.end(),
                                                 {builtin::param{name}, std::move(s), {}});
        }
        else if(ins->name() == "@outline")
        {
            auto s = ins->get_shape();
            copy_ins =
                impl->instructions.insert(impl->instructions.end(), {builtin::outline{s}, s, {}});
        }
        else
        {
            // retrieve its mapped input
            auto inputs = ins->inputs();
            // ensure all inputs have its corresponding copy instructions
            assert(std::all_of(
                inputs.begin(), inputs.end(), [&](auto i) { return ins_map.count(i) > 0; }));
            std::vector<instruction_ref> copy_inputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), copy_inputs.begin(), [&](auto i) {
                return ins_map[i];
            });
            copy_ins = add_instruction(ins->get_operator(), copy_inputs);
        }

        ins_map[ins] = copy_ins;
    }
}

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
    shape r     = compute_shape(op, args);
    auto result = impl->instructions.insert(ins, {op, r, std::move(args)});
    instruction::backreference(result);
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

    if(ins == std::prev(this->end()))
    {
        return replace_instruction(ins, op::identity{}, rep);
    }

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
    assert(std::all_of(first, last, [&](const instruction& ins) { return ins.outputs().empty(); }));
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

instruction_ref program::add_return(const std::vector<std::string>& names, std::vector<instruction_ref> args)
{
    auto result =
        insert_instruction(impl->instructions.end(), builtin::add_return{names}, std::move(args));
    return result;
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

std::vector<shape> program::get_output_shapes() const
{
    auto last_ins = impl->instructions.back();
    if(last_ins.name() == "add_return")
    {
        auto& output_ins = last_ins.inputs();
        std::vector<shape> output_shapes;
        std::transform(output_ins.begin(),
                       output_ins.end(),
                       std::back_inserter(output_shapes),
                       [](auto& ins) { return ins->get_shape(); });

        return output_shapes;
    }
    // The else branch is to provide backward compatibility
    else
    {
        return {last_ins.get_shape()};
    }
}

context& program::get_context() const { return impl->ctx; }

instruction_ref program::validate() const
{
    return std::find_if(impl->instructions.begin(),
                        impl->instructions.end(),
                        [&](const instruction& i) { return !i.valid(impl->instructions.begin()); });
}

void program::compile(const target& t, compile_options options)
{
    assert(this->validate() == impl->instructions.end());
    this->impl->ctx = t.get_context();
    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
        options.trace = tracer{std::cout};
    options.trace(*this);
    options.trace();
    run_passes(*this, t.get_passes(this->impl->ctx, options), options.trace);
    auto invalid = this->validate();
    if(invalid != impl->instructions.end())
    {
        auto index = std::distance(impl->instructions.begin(), invalid);
        MIGRAPHX_THROW("Invalid program from compilation at instruction " + std::to_string(index));
    }
    this->finalize();
}

void program::finalize()
{
    for(auto ins : iterator_for(*this))
    {
        ins->finalize(this->impl->ctx);
    }
}

template <class F>
std::vector<argument> generic_eval(const program& p,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   F trace)
{
    assert(p.validate() == p.end());
    std::unordered_map<instruction_ref, argument> ins_results;
    ins_results.reserve(p.size() * 2);
    std::vector<argument> values;
    values.reserve(16);
    for(auto ins : iterator_for(p))
    {
        const auto& name = ins->name();
        if(name == "@literal")
        {
            ins_results.emplace(ins, trace(ins, [&] { return ins->get_literal().get_argument(); }));
        }
        else if(name == "@param")
        {
            ins_results.emplace(
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
            ins_results.emplace(ins, trace(ins, [&] {
                                    return argument{ins->get_shape(), nullptr};
                                }));
        }
        else if(name == "add_return")
        {
            std::vector<argument> prog_outputs;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(prog_outputs),
                           [&](instruction_ref i) {
                               assert(ins_results.find(i) != ins_results.end());
                               return ins_results[i];
                           });

            return prog_outputs;
        }
        else
        {
            values.resize(ins->inputs().size());
            std::transform(
                ins->inputs().begin(), ins->inputs().end(), values.begin(), [&](instruction_ref i) {
                    assert(ins_results.find(i) != ins_results.end());
                    return ins_results[i];
                });
            ins_results.emplace(ins, trace(ins, [&] {
                                    return ins->get_operator().compute(
                                        ctx, ins->get_shape(), values);
                                }));
        }
        assert(ins_results.find(ins) != ins_results.end());
    }

    return {ins_results.at(std::prev(p.end()))};
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
            if(trace_level > 1 and ins->name().front() != '@' and ins->name() != "load" and
               ins->name() != "add_return")
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

    print_program(*this, [&](auto ins, const auto& names) {
        print_instruction(std::cout, ins, names);
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
    if(ins == this->end())
    {
        std::cout << "End instruction" << std::endl;
        return;
    }
    if(not has_instruction(ins))
    {
        std::cout << "Instruction not part of program" << std::endl;
        return;
    }
    std::stringstream ss;
    print_program(*this, [&](auto x, const auto& names) {
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

static std::string enclose_name(const std::string& name)
{
    return '"' + replace_string(name, "\"", "\\\"") + '"';
}

void program::print_graph(std::ostream& os, bool brief) const
{
    os << "digraph {" << std::endl;
    os << "\trankdir=LR;" << std::endl;
    print_program(*this, [&](auto ins, const auto& names) {
        std::string label;
        if(brief)
            label = ins->name();
        else
            label = to_string(ins->get_operator());
        os << "\t" << enclose_name(names.at(ins)) << "[label=" << enclose_name(label) << "]";
        os << ";" << std::endl;
        if(!ins->inputs().empty())
        {
            for(auto&& arg : ins->inputs())
            {
                os << "\t" << enclose_name(names.at(arg)) << " -> " << enclose_name(names.at(ins));
                if(not brief)
                    os << "[label=" << enclose_name(to_string(ins->get_shape())) << "]";
                os << ";" << std::endl;
            }
        }
    });
    os << "}" << std::endl;
}

static std::string cpp_var_name(const std::string& name)
{
    return "m" + replace_string(name, "@", "x");
}

static std::string cpp_op_var(const std::string& name, instruction_ref ins)
{
    return replace_string(name, "@", ins->name());
}

static void print_op_attributes(std::ostream& os, const std::string& name, const operation& op)
{
    std::string x = to_string(op);
    if(contains(x, "["))
    {
        auto start                 = x.find('[');
        auto end                   = x.find(']');
        std::string attribute_text = x.substr(start + 1, end - start - 1);
        std::vector<std::string> attributes;
        for(auto&& attribute : split_string(attribute_text, ','))
        {
            if(contains(attribute, '='))
                attributes.push_back(attribute);
            else
                attributes.back() += "," + attribute;
        }
        for(auto&& attribute : attributes)
        {
            auto p     = split_string(attribute, '=');
            auto key   = p.front();
            auto value = p.back();
            if(contains({"bn_mode", "padding_mode"}, key))
                continue;
            if(key == "mode")
                value = enclose_name(trim(value));
            os << name << "." << key << " = " << value << ";" << std::endl;
        }
    }
}

static void print_cpp_shape(std::ostream& os, const migraphx::shape& s)
{
    os << "migraphx::shape{migraphx::shape::" << s.type_string();
    os << ", {" << to_string_range(s.lens()) << "}";
    if(not s.standard())
        os << ", {" << to_string_range(s.strides()) << "}";
    os << "}";
}

void program::print_cpp(std::ostream& os) const
{
    os << "migraphx::program p;" << std::endl;
    // cppcheck-suppress variableScope
    unsigned long seed = 0;
    print_program(*this, [&](auto ins, const auto& names) {
        auto op = cpp_op_var(names.at(ins), ins);
        if(ins->name().front() != '@')
        {
            os << "migraphx::op::" << ins->name() << " " << op << ";" << std::endl;
            print_op_attributes(os, op, ins->get_operator());
        }
        os << "auto " << cpp_var_name(names.at(ins)) << " = ";
        if(ins->name() == "@literal")
        {
            os << "p.add_literal(";
            bool use_abs = false;
            ins->get_literal().visit([&](auto v) {
                use_abs = std::none_of(v.begin(), v.end(), [](auto x) { return x < 0; });
            });
            if(use_abs)
                os << "migraphx::abs(";
            os << "migraphx::generate_literal(";
            print_cpp_shape(os, ins->get_shape());
            os << ", " << seed << ")";
            if(use_abs)
                os << ")";
            os << ");" << std::endl;
            seed++;
        }
        else if(ins->name() == "@param")
        {
            std::string name = any_cast<builtin::param>(ins->get_operator()).parameter;
            os << "p.add_parameter(" << enclose_name(name) << ",";
            print_cpp_shape(os, ins->get_shape());
            os << ");" << std::endl;
        }
        else
        {
            os << "p.add_instruction(" << op;
            for(auto input : ins->inputs())
            {
                os << ", " << cpp_var_name(names.at(input));
            }
            os << ");" << std::endl;
        }
    });
}

void program::dry_run(std::unordered_map<std::string, argument> params) const
{
    auto& ctx = this->impl->ctx;
    generic_eval(*this, ctx, std::move(params), [](auto&&...) { return argument{}; });
}

void program::annotate(std::ostream& os, std::function<void(instruction_ref)> a) const
{
    print_program(*this, [&](auto ins, const auto& names) {
        print_instruction(os, ins, names);
        a(ins);
        os << std::endl;
    });
}

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    print_program(p, [&](auto ins, const auto& names) {
        print_instruction(os, ins, names);
        os << std::endl;
    });
    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
