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
#include <migraphx/make_op.hpp>
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
    // all the modules are store in the depth-first order
    std::list<module> modules;
    context ctx;
    std::string target_name;
};

program::program() : impl(std::make_unique<program_impl>()) { impl->modules.push_back({"main"}); }

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
    impl->modules     = p.impl->modules;

    // build a map from old ins to new ins
    // Build a map from old module to new module
    std::unordered_map<module_ref, module_ref> mod_map;
    std::transform(impl->modules.begin(),
                   impl->modules.end(),
                   p.impl->modules.begin(),
                   std::inserter(mod_map, mod_map.begin()),
                   [](auto&& x, auto&& y) { return std::make_pair(&y, &x); });

    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto&& pp : mod_map)
    {
        auto old_ins = iterator_for(*pp.first);
        auto new_ins = iterator_for(*pp.second);
        std::transform(old_ins.begin(),
                       old_ins.end(),
                       new_ins.begin(),
                       std::inserter(ins_map, ins_map.begin()),
                       [](auto x, auto y) { return std::make_pair(x, y); });
    }

    // Update all references from all modules
    for(auto&& mp : impl->modules)
    {
        for(auto ins : iterator_for(mp))
            instruction::replace_refs(ins, ins_map, mod_map);
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

    auto mods = this->get_modules();
    std::reverse(mods.begin(), mods.end());
    auto pass_options = options;

    for(const auto& mod : mods)
    {
        // submodules do not use offload_copy
        pass_options.offload_copy = ((mod->name() == "main") ? options.offload_copy : false);
        auto&& passes             = t.get_passes(this->impl->ctx, pass_options);

        assert(mod->validate() == mod->end());
        run_passes(*mod, passes, options.trace);
        auto invalid = mod->validate();
        if(invalid != mod->end())
        {
            auto index = std::distance(mod->begin(), invalid);
            MIGRAPHX_THROW("Invalid module " + mod->name() + " from compilation at instruction " +
                           std::to_string(index));
        }
        mod->finalize(this->impl->ctx);
    }
}

void program::finalize()
{
    auto* mm = this->get_main_module();
    mm->finalize(this->impl->ctx);
}

template <class F>
std::vector<argument> generic_eval(const module* mod,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   std::unordered_map<instruction_ref, argument> results,
                                   F trace)
{
    assert(mod->validate() == mod->end());
    results.reserve(mod->size() * 2);
    std::vector<argument> values;
    values.reserve(16);
    for(auto ins : iterator_for(*mod))
    {
        const auto& name = ins->name();
        if(name == "@literal")
        {
            results.emplace(ins,
                            trace(ins, mod, [&] { return ins->get_literal().get_argument(); }));
        }
        else if(name == "@param")
        {
            results.emplace(
                ins, trace(ins, mod, [&] {
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
            results.emplace(ins, trace(ins, mod, [&] {
                                return argument{ins->get_shape(), nullptr};
                            }));
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

            const auto& mod_args = ins->module_inputs();
            if(not mod_args.empty())
                results.emplace(
                    ins, trace(ins, mod, [&] {
                        return ins->get_operator().compute(
                            values,
                            mod_args,
                            [&](module_ref smod,
                                const std::unordered_map<std::string, argument>& inputs) {
                                // wrap up parameters for sub_modules
                                const auto& param_names = smod->get_parameter_names();
                                parameter_map m;
                                for(const auto& nm : param_names)
                                    if(contains(inputs, nm))
                                        m[nm] = inputs.at(nm);
                                    else if(contains(params, nm))
                                        m[nm] = params.at(nm);
                                    else
                                        MIGRAPHX_THROW("Input " + nm + " parameter of module: \"" +
                                                       mod->name() + "\" not exist!");
                                return generic_eval(smod, ctx, m, results, trace);
                            });
                    }));
            else
            {
                results.emplace(ins, trace(ins, mod, [&] {
                                    return ins->normalized_operator().compute(
                                        ctx, ins->get_shape(), values);
                                }));
            }
        }
        assert(results.find(ins) != results.end());
    }

    return {results.at(std::prev(mod->end()))};
}

template <class F>
std::vector<argument> generic_eval(const program& p,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   F trace)
{
    const module* mm = p.get_main_module();
    return generic_eval(mm, ctx, params, {}, trace);
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
        std::unordered_map<instruction_ref, std::string> names;
        return generic_eval(
            *this, ctx, std::move(params), [&](auto& ins, const module* mod, auto f) {
                ctx.finish();
                std::cout << "Run instruction: ";
                mod->debug_print(ins);
                auto result = check_context(f);
                ctx.finish();
                if(trace_level > 1 and ins->name().front() != '@' and ins->name() != "load")
                    std::cout << "Ouput: " << result << std::endl;
                return result;
            });
    }
    else
    {
        return generic_eval(*this, ctx, std::move(params), [&](auto&, const module*, auto f) {
            return check_context(f);
        });
    }
}

const int program_file_version = 4;

value program::to_value() const
{
    value result;
    result["version"] = program_file_version;
    result["target"]  = this->impl->target_name;
    if(not this->impl->target_name.empty())
        result["context"] = this->impl->ctx.to_value();

    value module_vals = value::array{};
    std::unordered_map<instruction_ref, std::string> names;
    for(auto& mod : this->impl->modules)
    {
        value mod_val;
        value nodes;
        mod_val["name"] = mod.name();
        names           = mod.print(
            [&](auto ins, auto ins_names) {
                value node;
                node["output"]     = ins_names.at(ins);
                node["name"]       = ins->name();
                node["shape"]      = migraphx::to_value(ins->get_shape());
                node["normalized"] = ins->is_normalized();
                if(ins->name() == "@literal")
                    node["literal"] = migraphx::to_value(ins->get_literal());
                node["operator"] = ins->get_operator().to_value();
                std::vector<std::string> inputs;
                std::transform(ins->inputs().begin(),
                               ins->inputs().end(),
                               std::back_inserter(inputs),
                               [&](auto i) {
                                   assert(contains(ins_names, i));
                                   return ins_names.at(i);
                               });
                node["inputs"]   = inputs;
                auto module_args = ins->module_inputs();
                if(not module_args.empty())
                {
                    std::vector<std::string> module_inputs;
                    std::transform(module_args.begin(),
                                   module_args.end(),
                                   std::back_inserter(module_inputs),
                                   [&](auto mod_ref) { return mod_ref->name(); });
                    node["module_inputs"] = module_inputs;
                }

                nodes.push_back(node);
            },
            names);
        mod_val["nodes"] = nodes;

        module_vals.push_back(mod_val);
    }

    result["modules"] = module_vals;

    return result;
}

static void mod_from_val(module_ref mod,
                         const value& v,
                         std::unordered_map<std::string, instruction_ref>& instructions,
                         const std::unordered_map<std::string, module_ref>& map_mods)
{
    const auto* it = std::find_if(v.begin(), v.end(), [&](auto& mv) {
        return mv.at("name").template to<std::string>() == mod->name();
    });
    assert(it != v.end());

    const auto& module_val = *it;
    for(const value& node : module_val.at("nodes"))
    {
        instruction_ref output;
        auto name       = node.at("name").to<std::string>();
        auto fields     = node.at("operator");
        auto normalized = node.at("normalized").to<bool>();

        if(name == "@param")
        {
            output = mod->add_parameter(fields["parameter"].to<std::string>(),
                                        migraphx::from_value<shape>(node.at("shape")));
        }
        else if(name == "@literal")
        {
            output = mod->add_literal(migraphx::from_value<literal>(node.at("literal")));
        }
        else
        {
            auto op = make_op(name, fields);
            std::vector<instruction_ref> inputs;
            std::transform(node.at("inputs").begin(),
                           node.at("inputs").end(),
                           std::back_inserter(inputs),
                           [&](const value& i) {
                               auto i_name = i.to<std::string>();
                               assert(contains(instructions, i_name));
                               return instructions.at(i_name);
                           });

            std::vector<module_ref> module_inputs;
            if(node.contains("module_inputs"))
            {
                std::transform(node.at("module_inputs").begin(),
                               node.at("module_inputs").end(),
                               std::back_inserter(module_inputs),
                               [&](const value& i) { return map_mods.at(i.to<std::string>()); });

                for(auto& smod : module_inputs)
                {
                    mod_from_val(smod, v, instructions, map_mods);
                }
            }

            if(name == "@return")
            {
                output = mod->add_return(inputs);
            }
            else if(module_inputs.empty())
            {
                output = mod->add_instruction(op, inputs);
            }
            else
            {
                output = mod->add_instruction(op, inputs, module_inputs);
            }
        }
        output->set_normalized(normalized);
        instructions[node.at("output").to<std::string>()] = output;
    }
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

    auto module_vals = v.at("modules");
    std::unordered_map<std::string, module_ref> map_mods;
    for(const auto& vv : module_vals)
    {
        const auto& name = vv.at("name").to<std::string>();
        if(name == "main")
            continue;
        impl->modules.push_back({name});
        map_mods[name] = &impl->modules.back();
    }

    std::unordered_map<std::string, instruction_ref> map_insts;
    auto* mm = get_main_module();
    mod_from_val(mm, module_vals, map_insts, map_mods);

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
    generic_eval(*this, ctx, params, [&](auto ins, const module*, auto) {
        ins_vec[ins].reserve(n);
        return argument{};
    });
    // Run and time each instruction
    for(std::size_t i = 0; i < n; i++)
    {
        generic_eval(*this, ctx, params, [&](auto ins, const module*, auto f) {
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

    std::unordered_map<instruction_ref, std::string> names;
    this->print(names, [&](auto ins, auto ins_names) {
        instruction::print(std::cout, ins, ins_names);

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
    std::unordered_map<instruction_ref, std::string> names;
    if(std::any_of(this->impl->modules.begin(), this->impl->modules.end(), [&](const auto& it) {
           return (it.end() == ins);
       }))
    {
        std::cout << "End instruction" << std::endl;
        return;
    }
    else if(std::none_of(this->impl->modules.begin(),
                         this->impl->modules.end(),
                         [&](const auto& it) { return it.has_instruction(ins); }))
    {
        std::cout << "Instruction not part of program" << std::endl;
        return;
    }

    std::stringstream ss;
    this->print(names, [&](auto x, auto ins_names) {
        if(x == ins)
        {
            instruction::print(std::cout, x, ins_names);
            std::cout << std::endl;
        }
    });
}

void program::print(
    std::unordered_map<instruction_ref, std::string>& names,
    const std::function<void(instruction_ref, std::unordered_map<instruction_ref, std::string>)>&
        print_func) const
{
    for(const auto& mod : this->impl->modules)
    {
        std::cout << mod.name() << ":" << std::endl;
        mod.print(print_func, names);
    }
}

void program::print_graph(std::ostream& os, bool brief) const
{
    const auto* mm = this->get_main_module();
    mm->print_graph(os, brief);
}

void program::print_cpp(std::ostream& os) const
{
    auto vec_modules = this->get_modules();
    std::unordered_map<instruction_ref, std::string> names;
    for(auto& mod : vec_modules)
    {
        os << "module: \"" << mod->name() << "\"" << std::endl;
        names = mod->print_cpp(os, names);
        os << std::endl;
    }
}

void program::dry_run(std::unordered_map<std::string, argument> params) const
{
    auto& ctx = this->impl->ctx;
    generic_eval(*this, ctx, std::move(params), [](auto&&...) { return argument{}; });
}

void program::annotate(std::ostream& os, const std::function<void(instruction_ref)>& a) const
{
    for(auto& mod : this->impl->modules)
    {
        std::cout << mod.name() << ":" << std::endl;
        mod.annotate(os, a);
    }
}

const module* program::get_module(const std::string& name) const
{
    auto it = std::find_if(
        impl->modules.begin(), impl->modules.end(), [&](auto& m) { return (m.name() == name); });
    if(it == impl->modules.end())
    {
        return nullptr;
    }

    return &(*it);
}

module* program::create_module(const std::string& name)
{
    auto it = impl->modules.insert(impl->modules.end(), {name});
    return &(*it);
}

module* program::get_module(const std::string& name)
{
    auto it = std::find_if(
        impl->modules.begin(), impl->modules.end(), [&](auto& m) { return (m.name() == name); });
    if(it == impl->modules.end())
    {
        return nullptr;
    }

    return &(*it);
}

module* program::get_main_module() { return get_module("main"); }

const module* program::get_main_module() const { return get_module("main"); }

std::vector<const module*> program::get_modules() const
{
    const module* mm = this->get_main_module();
    std::vector<const module*> vec_modules;
    vec_modules.push_back(mm);
    auto sub_modules = mm->get_sub_modules();
    vec_modules.insert(vec_modules.end(), sub_modules.begin(), sub_modules.end());

    return vec_modules;
}

std::vector<module*> program::get_modules()
{
    module* mm = this->get_main_module();
    std::vector<module*> vec_modules;
    vec_modules.push_back(mm);
    auto sub_modules = mm->get_sub_modules();
    vec_modules.insert(vec_modules.end(), sub_modules.begin(), sub_modules.end());

    return vec_modules;
}

program& program::sort()
{
    for(auto& mod : this->impl->modules)
    {
        mod.sort();
    }

    return *this;
}

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    auto vec_modules = p.get_modules();
    std::unordered_map<instruction_ref, std::string> names;
    for(auto& mod : vec_modules)
    {
        os << "module: \"" << mod->name() << "\"" << std::endl;
        names = mod->print(
            [&](auto ins, auto ins_names) {
                instruction::print(os, ins, ins_names);
                os << std::endl;
            },
            names);
        os << std::endl;
    }

    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
