/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/optimize_gather.hpp>
#include <migraphx/gather_optimizer.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/gather.hpp>
#include <migraphx/make_op.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_GATHER_OPTIMIZATION)

namespace {

/**
 * Checks if an instruction is a constant data source
 * Returns true for literal and param instructions
 */
bool is_constant_data(instruction_ref ins)
{
    if(ins == instruction_ref{})
        return false;
    
    auto name = ins->name();
    // Literals are always constant
    if(name == "@literal")
        return true;
    
    // Parameters can be constant if they're weights/embeddings (not batch inputs)
    // For now, we conservatively treat all parameters as potentially constant
    if(name == "@param")
        return true;
    
    return false;
}

/**
 * Analyzes a gather instruction and annotates it with optimization hints
 */
void analyze_and_annotate_gather(module& m, instruction_ref ins)
{
    auto op = any_cast<op::gather>(ins->get_operator());
    auto axis = op.axis;
    
    // Get input instructions
    auto inputs = ins->inputs();
    if(inputs.size() < 2)
        return;
    
    auto data_ins = inputs[0];
    auto indices_ins = inputs[1];
    
    auto data_shape = data_ins->get_shape();
    auto indices_shape = indices_ins->get_shape();
    auto output_shape = ins->get_shape();
    
    // Skip dynamic shapes for now
    if(data_shape.dynamic() || indices_shape.dynamic() || output_shape.dynamic())
        return;
    
    // Check if data input is constant
    bool data_is_constant = is_constant_data(data_ins);
    
    // Create shape vector for analysis
    std::vector<shape> shapes = {data_shape, indices_shape, output_shape};
    
    // Analyze and select optimal kernel
    auto analysis = analyze_gather(shapes, axis, data_is_constant);
    auto optimization = select_gather_optimization(analysis);
    auto kernel_name = get_gather_kernel_name(optimization);
    
    // Trace output if enabled
    if(enabled(MIGRAPHX_TRACE_GATHER_OPTIMIZATION{}))
    {
        std::cout << "Gather Optimization Analysis:\n";
        std::cout << "  Instruction: " << ins->name() << "\n";
        std::cout << "  Data source: " << data_ins->name() << " ";
        std::cout << (data_is_constant ? "(constant)" : "(variable)") << "\n";
        std::cout << "  Indices source: " << indices_ins->name() << "\n";
        std::cout << "  Output elements: " << analysis.num_elements << "\n";
        std::cout << "  Axis: " << analysis.axis << " ";
        std::cout << (analysis.is_innermost_axis ? "(innermost)" : "(not innermost)") << "\n";
        std::cout << "  Contiguous input: " << (analysis.is_contiguous_input ? "yes" : "no") << "\n";
        std::cout << "  Large gather: " << (analysis.is_large_gather ? "yes" : "no") << "\n";
        std::cout << "  Selected kernel: " << kernel_name << "\n";
        std::cout << std::endl;
    }
    
    // If data is constant, annotate the operation
    if(data_is_constant)
    {
        // Create new operation with constant data hint
        auto new_op_value = op.to_value();
        new_op_value["data_is_constant"] = true;
        
        // Create new gather operation with the annotated value
        auto new_op = make_op("gather", new_op_value);
        
        // Replace the instruction with annotated version
        m.replace_instruction(ins, new_op, inputs);
    }
}

} // anonymous namespace

void optimize_gather::apply(module& m) const
{
    // Iterate through all instructions
    for(auto ins : iterator_for(m))
    {
        // Find gather operations
        if(ins->name() == "gather")
        {
            analyze_and_annotate_gather(m, ins);
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

