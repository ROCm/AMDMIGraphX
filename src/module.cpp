/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <iterator>
#include <migraphx/module.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/iterator.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/json.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <set>
#include <utility>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_FINALIZE)

struct module_impl
{
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;
    std::unordered_set<instruction*> instruction_set;
    std::string name;
    uint32_t nparams = 0;
    bool bypass      = false;

    bool contains(instruction_ref ins) const
    {
        if(is_end(ins, instructions.end()))
            return false;
        return instruction_set.count(std::addressof(*ins)) > 0;
    }

    template <class... Ts>
    instruction_ref emplace(instruction_ref pos, Ts&&... xs)
    {
        // cppcheck-suppress redundantInitialization
        auto r = instructions.emplace(pos, std::forward<Ts>(xs)...);
        instruction_set.insert(std::addressof(*r));
        return r;
    }
    instruction_ref insert(instruction_ref pos, const instruction& ins)
    {
        return emplace(pos, ins);
    }

    void clear()
    {
        instructions.clear();
        instruction_set.clear();
        nparams = 0;
    }

    void push_front(const instruction& ins) { insert(instructions.begin(), ins); }

    void push_back(const instruction& ins) { insert(instructions.end(), ins); }

    template <class... Ts>
    void emplace_front(Ts&&... xs)
    {
        emplace(instructions.begin(), std::forward<Ts>(xs)...);
    }

    template <class... Ts>
    void emplace_back(Ts&&... xs)
    {
        emplace(instructions.end(), std::forward<Ts>(xs)...);
    }

    instruction_ref erase(instruction_ref pos)
    {
        instruction_set.erase(std::addressof(*pos));
        return instructions.erase(pos);
    }

    instruction_ref erase(instruction_ref start, instruction_ref last)
    {
        std::for_each(start, last, [&](auto& ins) { instruction_set.erase(std::addressof(ins)); });
        return instructions.erase(start, last);
    }
};

const operation& get_operation(instruction_ref ins) { return ins->get_operator(); }

module::module(const std::string& name) : impl(std::make_unique<module_impl>())
{
    impl->name = name;
}

module::module(module&&) noexcept = default;
module::~module() noexcept        = default;

// copy constructor
module::module(const module& m) { assign(m); }

// copy assignment operator
module& module::operator=(module m)
{
    std::swap(m.impl, this->impl);
    return *this;
}

std::string module::name() const { return impl->name; }

bool module::bypass() const { return impl->bypass; }
void module::set_bypass(bool b) { impl->bypass = b; }

void module::assign(const module& m)
{
    // copy the impl
    if(!impl)
        impl = std::make_unique<module_impl>();
    *impl = *m.impl;

    // clear instructions
    if(!impl->instructions.empty())
    {
        impl->clear();
    }

    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto ins : iterator_for(m))
    {
        instruction_ref copy_ins{};
        if(ins->name() == "@literal")
        {
            auto l   = ins->get_literal();
            copy_ins = impl->insert(impl->instructions.end(), instruction{l});
        }
        else if(ins->name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(ins->get_operator()).parameter;
            auto order  = any_cast<builtin::param>(ins->get_operator()).order;
            auto s      = ins->get_shape();
            copy_ins    = impl->insert(impl->instructions.end(),
                                    {builtin::param{name, order}, std::move(s), {}});
        }
        else if(ins->name() == "@outline")
        {
            auto s   = ins->get_shape();
            copy_ins = impl->insert(impl->instructions.end(), {builtin::outline{s}, s, {}});
        }
        else
        {
            // if there are sub_module inputs, need to make a copy of the submodule
            auto module_args = ins->module_inputs();
            // retrieve its mapped input
            auto inputs = ins->inputs();
            std::vector<instruction_ref> copy_inputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), copy_inputs.begin(), [&](auto i) {
                return contains(ins_map, i) ? ins_map[i] : i;
            });
            if(ins->name() == "@return")
            {
                copy_ins = add_return(copy_inputs);
            }
            else
            {
                copy_ins = add_instruction(ins->get_operator(), copy_inputs, module_args);
            }
        }

        ins_map[ins] = copy_ins;
    }
}

template <class Range>
static std::vector<instruction_ref>
insert_generic_instructions(module& m,
                            instruction_ref ins,
                            Range&& instructions,
                            std::unordered_map<instruction_ref, instruction_ref> map_ins)
{
    assert(m.has_instruction(ins) or is_end(ins, m.end()));
    std::vector<instruction_ref> mod_outputs;
    instruction_ref last;
    for(instruction_ref sins : instructions)
    {
        last = sins;
        if(contains(map_ins, sins))
            continue;
        instruction_ref copy_ins;
        if(sins->name() == "@literal")
        {
            auto l   = sins->get_literal();
            copy_ins = m.add_literal(l);
        }
        else if(sins->name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(sins->get_operator()).parameter;
            auto s      = sins->get_shape();
            copy_ins    = m.add_parameter(name, s);
        }
        else if(sins->name() == "@outline")
        {
            auto s   = sins->get_shape();
            copy_ins = m.add_outline(s);
        }
        else
        {
            auto mod_args = sins->module_inputs();
            auto inputs   = sins->inputs();
            std::vector<instruction_ref> copy_inputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), copy_inputs.begin(), [&](auto i) {
                return contains(map_ins, i) ? map_ins[i] : i;
            });

            if(sins->name() == "@return")
            {
                mod_outputs = copy_inputs;
                break;
            }

            copy_ins = m.insert_instruction(ins, sins->get_operator(), copy_inputs, mod_args);
        }
        map_ins[sins] = copy_ins;
    }
    if(mod_outputs.empty() and instructions.begin() != instructions.end())
        mod_outputs = {map_ins.at(last)};
    return mod_outputs;
}

instruction_ref module::add_instruction(const operation& op, std::vector<instruction_ref> args)
{
    return insert_instruction(impl->instructions.end(), op, std::move(args));
}
instruction_ref module::insert_instruction(instruction_ref ins,
                                           const operation& op,
                                           std::vector<instruction_ref> args)
{
    assert(has_instruction(ins) or is_end(ins, this->end()));
    assert(not starts_with(op.name(), "@"));
    shape r     = compute_shape(op, args);
    auto result = impl->insert(ins, {op, r, std::move(args)});
    instruction::backreference(result);
    assert(result->valid(begin()));
    return result;
}

instruction_ref module::add_instruction(const operation& op,
                                        std::vector<instruction_ref> args,
                                        std::vector<module_ref> module_args)
{
    return insert_instruction(
        impl->instructions.end(), op, std::move(args), std::move(module_args));
}

instruction_ref module::insert_instruction(instruction_ref ins,
                                           const operation& op,
                                           std::vector<instruction_ref> args,
                                           std::vector<module_ref> module_args)
{
    assert(has_instruction(ins) or is_end(ins, this->end()));
    assert(not starts_with(op.name(), "@"));
    auto out_shape = compute_shape(op, args, module_args);
    auto result    = impl->insert(ins, {op, out_shape, std::move(args), std::move(module_args)});
    instruction::backreference(result);
    assert(result->valid(begin()));
    return result;
}

instruction_ref module::replace_instruction(instruction_ref ins,
                                            const operation& op,
                                            std::vector<instruction_ref> args) MIGRAPHX_TIDY_CONST
{
    assert(has_instruction(ins));
    assert(not starts_with(op.name(), "@"));

    shape r = compute_shape(op, args);
    instruction::replace(ins, op, r, std::move(args));
    assert(ins->valid(begin()));
    return ins;
}

instruction_ref module::replace_instruction(instruction_ref ins,
                                            const operation& op,
                                            std::vector<instruction_ref> args,
                                            std::vector<module_ref> module_args) MIGRAPHX_TIDY_CONST
{
    assert(has_instruction(ins));
    assert(not starts_with(op.name(), "@"));
    auto out_shape = compute_shape(op, args, module_args);
    instruction::replace(ins, op, out_shape, std::move(args), std::move(module_args));
    assert(ins->valid(begin()));
    return ins;
}

instruction_ref module::replace_instruction(instruction_ref ins, instruction_ref rep)
{
    assert(has_instruction(ins));
    assert(has_instruction(rep));
    assert(ins != rep);

    if(ins == std::prev(this->end()))
    {
        return replace_instruction(ins, make_op("identity"), rep);
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

instruction_ref module::remove_instruction(instruction_ref ins)
{
    assert(has_instruction(ins));
    assert(ins->outputs().empty());
    ins->clear_arguments();
    return impl->erase(ins);
}

instruction_ref module::remove_instructions(instruction_ref first, instruction_ref last)
{
    if(first == last)
        return first;
    // TODO: Check every element
    assert(has_instruction(first));
    std::for_each(first, last, [&](instruction& ins) { ins.clear_arguments(); });
    assert(std::all_of(first, last, [&](const instruction& ins) { return ins.outputs().empty(); }));
    return impl->erase(first, last);
}

instruction_ref module::move_instruction(instruction_ref src, instruction_ref dst)
{
    assert(has_instruction(src));
    assert(has_instruction(dst) or is_end(dst, this->end()));
    impl->instructions.splice(dst, impl->instructions, src);
    return src;
}

instruction_ref module::move_instructions(instruction_ref src, instruction_ref dst)
{
    this->move_instruction(src, dst);
    for(auto ins : src->inputs())
        this->move_instruction(ins, src);
    return src;
}

std::vector<instruction_ref>
module::add_instructions(const std::vector<instruction_ref>& instructions,
                         std::unordered_map<instruction_ref, instruction_ref> map_ins)
{
    return this->insert_instructions(this->end(), instructions, std::move(map_ins));
}

std::vector<instruction_ref>
module::add_instructions(const_module_ref m,
                         std::unordered_map<instruction_ref, instruction_ref> map_ins)
{
    return this->insert_instructions(this->end(), m, std::move(map_ins));
}

std::vector<instruction_ref>
module::add_instructions(instruction_ref start,
                         instruction_ref last,
                         std::unordered_map<instruction_ref, instruction_ref> map_ins)
{
    return this->insert_instructions(this->end(), start, last, std::move(map_ins));
}

std::vector<instruction_ref>
module::insert_instructions(instruction_ref ins,
                            const std::vector<instruction_ref>& instructions,
                            std::unordered_map<instruction_ref, instruction_ref> map_ins)
{
    return insert_generic_instructions(*this, ins, instructions, std::move(map_ins));
}

std::vector<instruction_ref>
module::insert_instructions(instruction_ref ins,
                            const_module_ref m,
                            std::unordered_map<instruction_ref, instruction_ref> map_ins)
{
    return insert_generic_instructions(*this, ins, iterator_for(*m), std::move(map_ins));
}

std::vector<instruction_ref>
module::insert_instructions(instruction_ref ins,
                            instruction_ref start,
                            instruction_ref last,
                            std::unordered_map<instruction_ref, instruction_ref> map_ins)
{
    auto r = range(start, last);
    return insert_generic_instructions(*this, ins, iterator_for(r), std::move(map_ins));
}

instruction_ref module::add_literal(literal l)
{
    impl->emplace_front(std::move(l));
    return impl->instructions.begin();
}

instruction_ref module::add_outline(const shape& s)
{
    impl->push_front({builtin::outline{s}, s, {}});
    return impl->instructions.begin();
}

instruction_ref module::add_parameter(std::string name, shape s)
{
    assert(get_parameter_shape(name) == shape{});
    impl->push_front({builtin::param{std::move(name), impl->nparams}, std::move(s), {}});
    impl->nparams++;
    return impl->instructions.begin();
}

instruction_ref module::add_return(std::vector<instruction_ref> args)
{
    impl->push_back({builtin::returns{}, {}, std::move(args)});
    auto result = std::prev(impl->instructions.end());
    instruction::backreference(result);
    assert(result->valid(begin()));

    return result;
}

instruction_ref module::replace_return(std::vector<instruction_ref> args)
{
    auto last = std::prev(this->end());
    // If there is no return then add a return
    if(last->name() != "@return")
        return this->add_return(args);

    shape r = compute_shape(last->get_operator(), args);
    instruction::replace(last, last->get_operator(), r, std::move(args));
    assert(last->valid(begin()));

    return last;
}

shape module::get_parameter_shape(std::string name) const
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

std::vector<std::string> module::get_parameter_names() const
{
    std::vector<std::string> result;
    std::vector<builtin::param> params;
    for(auto&& ins : impl->instructions)
    {
        if(ins.name() == "@param")
        {
            auto&& param = any_cast<builtin::param>(ins.get_operator());
            params.push_back(param);
        }
    }
    std::stable_sort(
        params.begin(), params.end(), by(std::less<>{}, [](auto&& p) { return p.order; }));
    std::transform(params.begin(), params.end(), std::back_inserter(result), [&](auto&& p) {
        return p.parameter;
    });
    return result;
}

instruction_ref module::get_parameter(std::string name) const
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

std::unordered_map<std::string, shape> module::get_parameter_shapes() const
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

bool module::has_instruction(instruction_ref ins) const { return impl->contains(ins); }

std::size_t module::size() const { return impl->instructions.size(); }
instruction_ref module::begin() const { return impl->instructions.begin(); }
instruction_ref module::end() const { return impl->instructions.end(); }

std::vector<shape> module::get_output_shapes() const
{
    if(impl->instructions.empty())
        return {};
    auto last_ins = impl->instructions.back();
    if(last_ins.name() == "@return")
    {
        const auto& output_ins = last_ins.inputs();
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

instruction_ref module::validate() const
{
    return std::find_if(
        impl->instructions.begin(), impl->instructions.end(), [&](const instruction& i) {
            auto inputs      = i.inputs();
            bool check_order = std::all_of(
                inputs.begin(), inputs.end(), [&](auto in) { return has_instruction(in); });
            return !i.valid(impl->instructions.begin(), check_order);
        });
}

bool is_borrowed(instruction_ref ins)
{
    auto alias = instruction::get_output_alias(ins, true);
    if(alias == ins)
        return false;
    lifetime l = alias->get_operator().get_lifetime();
    if(l == lifetime::borrow)
        return true;
    return is_borrowed(alias);
}

bool is_global(instruction_ref ins)
{
    const auto& op = instruction::get_output_alias(ins)->get_operator();
    return op.name() == "@param" or op.get_lifetime() == lifetime::global;
}

bool is_dangling(instruction_ref ins) { return not is_global(ins) and is_borrowed(ins); }

instruction_ref module::find_dangling_reference() const
{
    auto last = std::prev(end());
    if(last->name() == "@return")
    {
        auto dangling = std::find_if(
            last->inputs().begin(), last->inputs().end(), [](auto x) { return is_dangling(x); });
        if(dangling != last->inputs().end())
            return *dangling;
    }
    else if(is_dangling(last))
    {
        return last;
    }
    return end();
}

void module::finalize(context& ctx)
{
    const bool trace = enabled(MIGRAPHX_TRACE_FINALIZE{});
    for(auto ins : iterator_for(*this))
    {
        if(trace)
        {
            std::cout << "Finalize: ";
            this->debug_print(ins);
        }
        ins->finalize(ctx);
        for(const auto& smod : ins->module_inputs())
        {
            smod->finalize(ctx);
        }
    }

    // Warn when an instruction is not normalized
    auto ins = std::find_if(begin(), end(), [](auto& i) { return i.need_normalization(); });
    if(ins != end())
        std::cerr << "WARNING: Instruction needs normalization, performance may be affected."
                  << std::endl;
}

void module::debug_print() const { std::cout << *this << std::endl; }

void module::debug_print(instruction_ref ins,
                         std::unordered_map<instruction_ref, std::string>& names) const
{
    if(is_end(ins, this->end()))
    {
        std::cout << "End instruction" << std::endl;
        return;
    }
    if(not has_instruction(ins))
    {
        std::cout << "Instruction not part of module" << std::endl;
        return;
    }
    std::stringstream ss;
    names = this->print(
        [&](auto x, auto ins_names) {
            if(x == ins)
            {
                instruction::print(std::cout, x, ins_names);
                std::cout << std::endl;
            }
        },
        names);
}

void module::debug_print(instruction_ref ins) const
{
    std::unordered_map<instruction_ref, std::string> names;
    this->debug_print(ins, names);
}

void module::debug_print(const std::vector<instruction_ref>& inss) const
{
    for(auto ins : inss)
        this->debug_print(ins);
    std::cout << std::endl;
}

std::unordered_map<instruction_ref, std::string> module::print(
    const std::function<void(instruction_ref,
                             const std::unordered_map<instruction_ref, std::string>&)>& print_func,
    std::unordered_map<instruction_ref, std::string> names) const
{
    int count = 0;
    for(auto ins : iterator_for(*this))
    {
        std::string var_name;
        if(ins->name() == "@param")
        {
            var_name = any_cast<builtin::param>(ins->get_operator()).parameter;
        }
        else
        {
            var_name = this->name();
            var_name.append((this->name().empty() ? "@" : ":@"));
            var_name.append(std::to_string(count));
        }
        // count every instruction so index matches loc in the printout program
        count++;
        names.emplace(ins, var_name);

        print_func(ins, names);
    }
    return names;
}

void module::print(const std::function<
                   void(instruction_ref, const std::unordered_map<instruction_ref, std::string>&)>&
                       print_func) const
{
    this->print(print_func, {});
}

static std::string enclose_name(const std::string& name)
{
    return '"' + replace_string(name, "\"", "\\\"") + '"';
}

void module::print_graph(std::ostream& os, bool brief) const
{
    os << "digraph {" << std::endl;
    os << "\trankdir=LR;" << std::endl;
    this->print([&](auto ins, auto ins_names) {
        std::string label;
        if(brief)
            label = ins->name();
        else
            label = to_string(ins->get_operator());
        os << "\t" << enclose_name(ins_names.at(ins)) << "[label=" << enclose_name(label) << "]";
        os << ";" << std::endl;
        if(!ins->inputs().empty())
        {
            for(auto&& arg : ins->inputs())
            {
                os << "\t" << enclose_name(ins_names.at(arg)) << " -> "
                   << enclose_name(ins_names.at(ins));
                if(not brief)
                    os << "[label=" << enclose_name(to_string(ins->get_shape())) << "]";
                os << ";" << std::endl;
            }
        }
    });
    os << "}" << std::endl;
}

static std::string to_c_id(const std::string& name, char rep = '_')
{
    std::string id = transform_string(name, [&](auto c) {
        if(with_char(::isalnum)(c) or c == '_')
            return c;
        return rep;
    });
    while(contains(id, "__"))
        replace_string_inplace(id, "__", "_");
    return id;
}

static std::string cpp_var_name(const std::string& name)
{
    return to_c_id("x_" + replace_string(name, ":", "_module_"));
}

static void print_make_op(std::ostream& os, const operation& op)
{
    os << "migraphx::make_op(" << enclose_name(op.name());
    auto v = op.to_value();
    if(not v.empty())
    {
        os << ", "
           << "migraphx::from_json_string(" << enclose_name(to_json_string(v)) << ")";
    }
    os << ")";
}

static void print_cpp_shape(std::ostream& os, const migraphx::shape& s)
{
    os << "migraphx::shape{migraphx::shape::" << s.type_string();
    os << ", {" << to_string_range(s.lens()) << "}";
    if(not s.standard())
        os << ", {" << to_string_range(s.strides()) << "}";
    os << "}";
}

std::unordered_map<instruction_ref, std::string>
module::print_cpp(std::ostream& os,
                  const std::string& mname,
                  std::unordered_map<instruction_ref, std::string> names) const
{
    // cppcheck-suppress variableScope
    unsigned long seed = names.size();
    auto last          = std::prev(this->end());
    names              = this->print(
        [&](auto ins, auto ins_names) {
            std::vector<std::string> input_vars;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(input_vars),
                           [&](auto input) { return cpp_var_name(ins_names.at(input)); });
            if(ins != last)
                os << "auto " << cpp_var_name(ins_names.at(ins)) << " = ";
            if(ins->name() == "@literal")
            {
                os << mname << "->add_literal(";
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
                os << mname << "->add_parameter(" << enclose_name(name) << ",";
                print_cpp_shape(os, ins->get_shape());
                os << ");" << std::endl;
            }
            else if(ins->name() == "@return")
            {
                os << mname << "->add_return({";
                os << join_strings(input_vars, ", ");
                os << "});" << std::endl;
            }
            else
            {
                assert(ins->name().front() != '@');
                os << mname << "->add_instruction(";
                print_make_op(os, ins->get_operator());
                os << ", " << join_strings(input_vars, ", ");
                os << ");" << std::endl;
            }
        },
        names);

    return names;
}

void module::print_cpp(std::ostream& os) const { this->print_cpp(os, this->name(), {}); }

void module::annotate(std::ostream& os, std::function<void(instruction_ref)> a) const
{
    this->print([&](auto ins, auto ins_names) {
        instruction::print(os, ins, ins_names);
        a(ins);
        os << std::endl;
    });
}

std::vector<module_ref> module::get_sub_modules(bool shallow) const
{
    std::vector<module_ref> vec_modules;
    for(auto ins : iterator_for(*this))
    {
        const auto& mod_args = ins->module_inputs();
        vec_modules.insert(vec_modules.end(), mod_args.begin(), mod_args.end());
        if(not shallow)
        {
            for(const auto& smod : mod_args)
            {
                auto sub_mods = smod->get_sub_modules();
                vec_modules.insert(vec_modules.end(), sub_mods.begin(), sub_mods.end());
            }
        }
    }

    return vec_modules;
}

module& module::sort()
{
    fix([&](auto self, auto ins) {
        this->move_instruction(ins, this->begin());
        for(auto child : ins->inputs())
        {
            if(!contains(this->impl->instructions, child))
            {
                continue;
            }
            self(child);
        }
    })(std::prev(this->end()));
    assert(this->validate() == this->end());
    return *this;
}

void module::calc_implicit_deps(const module& smod,
                                const module& pmod,
                                instruction_ref ins,
                                ins_dep_map& deps) const
{
    const auto& ins_inputs = ins->inputs();
    for(auto ii : iterator_for(smod))
    {
        const auto& ii_inputs = ii->inputs();
        for(auto iii : ii_inputs)
        {
            if(pmod.has_instruction(iii))
            {
                if(not contains(ins_inputs, iii))
                    deps[ins].insert(iii);
            }
        }

        const auto& mod_args = ii->module_inputs();
        if(not mod_args.empty())
        {
            for(const auto* ssmod : mod_args)
            {
                calc_implicit_deps(*ssmod, pmod, ins, deps);
            }
        }
    }
}

ins_dep_map module::calc_implicit_deps() const
{
    ins_dep_map mod_implicit_deps;
    for(auto ins : iterator_for(*this))
    {
        const auto& mod_args = ins->module_inputs();
        if(mod_args.empty())
        {
            continue;
        }

        for(const auto* mod : mod_args)
        {
            calc_implicit_deps(*mod, *this, ins, mod_implicit_deps);
        }
    }

    return mod_implicit_deps;
}

bool operator==(const module& x, const module& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const module& m)
{
    m.print([&](auto ins, auto ins_names) {
        instruction::print(os, ins, ins_names);
        os << std::endl;
    });

    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
