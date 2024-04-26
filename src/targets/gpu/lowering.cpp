/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <utility>
#include <functional>
#include <algorithm>
#include <map>

#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/program.hpp>

#include <migraphx/op/dot.hpp>
#include <migraphx/op/if_op.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/op/reshape_lazy.hpp>

#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/compiler.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct miopen_apply
{
    module* mod              = nullptr;
    module_pass_manager* mpm = nullptr;
    const lowering* pass     = nullptr;
    std::unordered_map<std::string, std::function<instruction_ref(instruction_ref)>> apply_map{};
    instruction_ref last{};
    bool offload_copy = false;
    bool compute_fp32 = false;

    context& get_context() const
    {
        assert(pass != nullptr);
        assert(pass->ctx != nullptr);
        return *pass->ctx;
    }

    void check_shape(shape x, instruction_ref i)
    {
        assert(x == i->get_shape());
        (void)x;
        (void)i;
    }

    void init()
    {
        assert(mod != nullptr);
        assert(pass != nullptr);

        compute_fp32 = get_compute_fp32_flag();
        offload_copy = (mod == mpm->get_root_module()) ? pass->offload_copy : false;

        add_generic_op("contiguous");
        add_extend_op("argmax");
        add_extend_op("argmin");
        add_extend_op("logsoftmax");
        add_extend_op("lrn");
        add_extend_op("multinomial");
        add_extend_op("nonzero");
        add_extend_op("pooling");
        add_extend_op("prefix_scan_sum");
        add_extend_op("reverse");
        add_extend_op("rnn_var_sl_last_output");
        add_extend_op("rnn_var_sl_shift_output");
        add_extend_op("rnn_var_sl_shift_sequence");
        add_extend_op("topk");

        add_convolution_op("convolution");
        add_convolution_op("convolution_backwards");
        add_convolution_op("quant_convolution");
        add_gemm_op<op::dot>("dot");
        add_gemm_op<op::quant_dot>("quant_dot");
        add_if_op();
        add_loop_op();
        add_neg_op();
        add_nms_op();
        add_select_module_op();
        add_reshape_lazy_op();
    }

    void copy_params() const
    {
        if(not offload_copy)
            return;

        for(auto ins : iterator_for(*mod))
        {
            if(ins->name() != "@param")
                continue;

            // parameter no outputs, no need to insert copy to gpu
            if(ins->outputs().empty())
                continue;

            auto pos = std::next(ins);
            auto a   = insert_allocation(pos, ins->get_shape());
            auto c   = mod->insert_instruction(pos, make_op("hip::copy_to_gpu"), ins, a);
            mod->replace_instruction(ins, c);
        }

        // return instruction
        auto ret = std::prev(mod->end());
        if(ret->name() == "@return")
        {
            const auto& inputs = ret->inputs();

            // each input of ret need to be copied from gpu to host, and replace
            // output with copy output
            for(const auto& in : inputs)
            {
                auto p_output = mod->insert_instruction(ret, make_op("hip::copy_from_gpu"), in);
                instruction::replace_argument(ret, in, p_output);
            }
        }
        // else branch to handle legacy program without the return instruction
        else
        {
            mod->add_instruction(make_op("hip::copy_from_gpu"), ret);
        }
    }

    void apply()
    {
        init();
        for(auto it = mod->begin(); it != mod->end(); it++)
        {
            auto s     = it->get_shape();
            auto attrs = it->get_operator().attributes();
            if(apply_map.count(it->name()) > 0)
            {
                check_shape(s, apply_map.at(it->name())(it));
            }
            else if(has_compiler_for(it->name()))
            {
                check_shape(s, insert_precompile_op(it));
            }
            else if(attrs.contains("target"))
            {
                check_shape(s, insert_custom_op(it, attrs));
            }
            if(attrs.contains("prefill"))
            {
                insert_fill(it, attrs.at("prefill"));
            }
        }
        copy_params();
    }

    void insert_fill(instruction_ref ins, value v) const
    {
        instruction_ref alloc = instruction::get_output_alias(ins, true);
        if(alloc == ins)
            return;
        auto fill = mod->insert_instruction(ins, make_op("hip::fill", {{"value", v}}), alloc);
        instruction::replace_argument(ins, alloc, fill);
    }

    instruction_ref insert_custom_op(instruction_ref ins, const value& attrs) const
    {
        const auto& custom_op = ins->get_operator();
        if(attrs.at("target") == "cpu")
        {
            auto s = ins->get_shape();
            std::vector<instruction_ref> cpu_inputs;
            auto inputs = ins->inputs();
            auto output = inputs.back();
            std::transform(
                inputs.begin(), inputs.end(), std::back_inserter(cpu_inputs), [&](auto in) {
                    return mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), in);
                });
            cpu_inputs.front() =
                mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_inputs);
            auto cpu_out = mod->insert_instruction(ins, custom_op, cpu_inputs);
            auto gpu_out =
                mod->insert_instruction(ins, make_op("hip::copy_to_gpu"), cpu_out, output);
            return mod->replace_instruction(ins, gpu_out);
        }
        return ins;
    }

    instruction_ref insert_precompile_op(instruction_ref ins) const
    {
        auto output                       = insert_allocation(ins, ins->get_shape());
        std::vector<instruction_ref> refs = ins->inputs();
        refs.push_back(output);

        return mod->replace_instruction(
            ins,
            make_op("gpu::precompile_op", {{"op", to_value(ins->get_operator())}}),
            refs,
            ins->module_inputs());
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s) const
    {
        return mod->insert_instruction(ins, make_op("allocate", {{"shape", to_value(s)}}));
    }

    template <typename Op>
    void add_gemm_op(const std::string& name)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            std::vector<instruction_ref> refs = ins->inputs();
            assert(refs.size() == 2);
            auto output = insert_allocation(ins, ins->get_shape());
            refs.push_back(output);
            return mod->replace_instruction(ins, rocblas_gemm<Op>{Op{}, 1, 0, compute_fp32}, refs);
        });
    }

    void add_convolution_op(const std::string& name)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            operation conv = make_op("gpu::" + name, {{"op", ins->get_operator().to_value()}});
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(ins,
                                            make_op("gpu::miopen_op", {{"op", to_value(conv)}}),
                                            ins->inputs().at(0),
                                            ins->inputs().at(1),
                                            output);
        });
    }

    // add_generic_op just constructs the operator with no fields whereas add_extend_op copies over
    // the fields Since it doesn't have fields its default constructed

    void add_generic_op(const std::string& name) { add_generic_op(name, "gpu::" + name); }

    void add_generic_op(const std::string& op_name, const std::string& gpu_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return mod->replace_instruction(ins, make_op(gpu_name), refs);
        });
    }

    void add_extend_op(const std::string& name) { add_extend_op(name, "gpu::" + name); }

    void add_extend_op(const std::string& op_name, const std::string& gpu_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto&& op                         = ins->get_operator();
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return mod->replace_instruction(ins, make_op(gpu_name, op.to_value()), refs);
        });
    }

    // use 0 - input to represent neg
    void add_neg_op()
    {
        apply_map.emplace("neg", [=](instruction_ref ins) {
            auto s = ins->get_shape();
            std::vector<float> zeros(s.elements(), 0.0f);
            auto l0     = mod->add_literal(literal(s, zeros));
            auto output = insert_allocation(ins, s);
            return mod->replace_instruction(
                ins, make_op("gpu::sub"), l0, ins->inputs().front(), output);
        });
    }

    // add input and output argument for the if operator
    void add_if_op()
    {
        apply_map.emplace("if", [=](instruction_ref ins) {
            std::vector<instruction_ref> inputs = ins->inputs();
            auto cpu_cond =
                mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), inputs.front());
            auto sync_cond = mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_cond);
            inputs.front() = sync_cond;

            return mod->replace_instruction(ins, ins->get_operator(), inputs, ins->module_inputs());
        });
    }

    // replace the loop operator with gpu_loop operator
    void add_loop_op()
    {
        apply_map.emplace("loop", [=](instruction_ref ins) {
            std::vector<instruction_ref> inputs = ins->inputs();
            // copy max_iter from gpu to cpu
            auto cpu_max_iter =
                mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), inputs.at(0));
            auto cpu_cond =
                mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), inputs.at(1));
            auto synced_max_iter =
                mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_max_iter, cpu_cond);
            inputs.at(0)     = synced_max_iter;
            inputs.at(1)     = cpu_cond;
            auto copy_inputs = inputs;
            std::transform(copy_inputs.begin(),
                           copy_inputs.end(),
                           std::back_inserter(inputs),
                           [&](auto in) { return insert_allocation(ins, in->get_shape()); });

            auto mod_args = ins->module_inputs();
            auto output   = insert_allocation(ins, ins->get_shape());

            const auto* sub_mod = mod_args.front();
            auto cond_out       = insert_allocation(ins, sub_mod->get_output_shapes().front());

            // add cond and mod outputs to the argument list
            inputs.push_back(cond_out);
            inputs.push_back(output);

            return mod->replace_instruction(
                ins, make_op("gpu::loop", ins->get_operator().to_value()), inputs, mod_args);
        });
    }

    void add_nms_op()
    {
        apply_map.emplace("nonmaxsuppression", [=](instruction_ref ins) {
            auto s      = ins->get_shape();
            auto output = insert_allocation(ins, s);
            std::vector<instruction_ref> cpu_inputs;
            auto inputs = ins->inputs();
            std::transform(
                inputs.begin(), inputs.end(), std::back_inserter(cpu_inputs), [&](auto in) {
                    return mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), in);
                });
            cpu_inputs.front() =
                mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_inputs);
            auto cpu_out = mod->insert_instruction(ins, ins->get_operator(), cpu_inputs);
            auto gpu_out =
                mod->insert_instruction(ins, make_op("hip::copy_to_gpu"), cpu_out, output);
            return mod->replace_instruction(ins, gpu_out);
        });
    }

    /**
     * Adds dynamic allocation for submodule output parameter.
     */
    void add_select_module_op()
    {
        apply_map.emplace("select_module", [=](instruction_ref ins) {
            auto s                              = ins->get_shape();
            auto output                         = insert_allocation(ins, s);
            std::vector<instruction_ref> inputs = ins->inputs();
            inputs.push_back(output);
            return mod->replace_instruction(ins, ins->get_operator(), inputs, ins->module_inputs());
        });
    }

    /**
     *  Adds reshape lazy to reshape ops that can be aliased instead of copied.
     *  `gpu::contiguous` are added before and after the reshape; these contiguous
     *  instructions can be removed by the eliminate_contiguous pass.
     */
    void add_reshape_lazy_op()
    {
        apply_map.emplace("reshape", [=](instruction_ref ins) {
            std::vector<instruction_ref> before_contiguous_args = ins->inputs();
            auto before_alloc = insert_allocation(ins, std::prev(ins)->get_shape());
            before_contiguous_args.push_back(before_alloc);
            auto before_contig =
                mod->insert_instruction(ins, make_op("gpu::contiguous"), {before_contiguous_args});

            auto new_lazy_reshape = mod->insert_instruction(
                ins,
                make_op("reshape_lazy", {{"dims", {ins->get_operator().to_value().at("dims")}}}),
                before_contig);

            std::vector<instruction_ref> after_contiguous_args = {new_lazy_reshape};
            auto after_alloc = insert_allocation(new_lazy_reshape, new_lazy_reshape->get_shape());
            after_contiguous_args.push_back(after_alloc);
            return mod->replace_instruction(ins, make_op("gpu::contiguous"), after_contiguous_args);
        });
    }
};

void lowering::apply(module_pass_manager& mpm) const
{
    miopen_apply{&mpm.get_module(), &mpm, this}.apply();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
