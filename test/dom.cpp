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
#include <migraphx/dom_info.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

bool strictly_dominates_self(const migraphx::dominator_info& dom, const migraphx::module& m)
{
    return migraphx::any_of(migraphx::iterator_for(m),
                            [&](auto ins) { return dom.strictly_dominate(ins, ins); });
}

// clang-format off
// ┌────┐            
// │ins1│            
// └┬───┘            
// ┌▽────────────┐   
// │ins2         │   
// └┬─────┬─────┬┘   
// ┌▽───┐┌▽───┐┌▽───┐
// │ins4││ins3││ins6│
// └┬───┘└┬───┘└────┘
// ┌▽─────▽┐         
// │ins5   │         
// └───────┘
// clang-format on
TEST_CASE(dom1)
{
    migraphx::module mm;
    auto ins1 = mm.add_parameter("entry", {migraphx::shape::float_type}); // ins1 -> ins2
    auto ins2 = mm.add_instruction(pass_op{}, ins1); // ins2 -> ins3, ins2 -> ins4, ins2 -> ins6
    auto ins3 = mm.add_instruction(pass_op{}, ins2); // ins3 -> ins5
    auto ins4 = mm.add_instruction(pass_op{}, ins2); // ins4 -> ins5
    auto ins5 = mm.add_instruction(pass_op{}, ins3, ins4);
    auto ins6 = mm.add_instruction(pass_op{}, ins2);

    auto dom = migraphx::compute_dominator(mm);
    CHECK(not strictly_dominates_self(dom, mm));
    // ins1
    CHECK(dom.strictly_dominate(ins1, ins2));
    CHECK(dom.strictly_dominate(ins1, ins3));
    CHECK(dom.strictly_dominate(ins1, ins4));
    CHECK(dom.strictly_dominate(ins1, ins5));
    CHECK(dom.strictly_dominate(ins1, ins6));
    // ins2
    CHECK(dom.strictly_dominate(ins2, ins3));
    CHECK(dom.strictly_dominate(ins2, ins4));
    CHECK(dom.strictly_dominate(ins2, ins5));
    CHECK(dom.strictly_dominate(ins2, ins6));

    CHECK(not dom.strictly_dominate(ins3, ins6));
    CHECK(not dom.strictly_dominate(ins4, ins6));
    CHECK(not dom.strictly_dominate(ins3, ins5));
    CHECK(not dom.strictly_dominate(ins4, ins5));
}

// clang-format off
// ┌────┐      
// │ins1│      
// └┬───┘      
// ┌▽───┐      
// │ins2│      
// └┬─┬─┘      
//  │┌▽───┐    
//  ││ins3│    
//  │└┬───┘    
//  │┌▽─────┐  
//  ││ins4  │  
//  │└┬───┬─┘  
// ┌▽─▽─┐┌▽───┐
// │ins5││ins6│
// └────┘└────┘
// clang-format on
TEST_CASE(dom2)
{
    migraphx::module mm;
    auto ins1 = mm.add_parameter("entry", {migraphx::shape::float_type}); // ins1 -> ins2
    auto ins2 = mm.add_instruction(pass_op{}, ins1); // ins2 -> ins3, ins2 -> ins5
    auto ins3 = mm.add_instruction(pass_op{}, ins2); // ins3 -> ins4
    auto ins4 = mm.add_instruction(pass_op{}, ins3); // ins4 -> ins5, ins4 -> ins6
    auto ins5 = mm.add_instruction(pass_op{}, ins2, ins4);
    auto ins6 = mm.add_instruction(pass_op{}, ins4);

    auto dom = migraphx::compute_dominator(mm);
    CHECK(not strictly_dominates_self(dom, mm));
    // ins1
    CHECK(dom.strictly_dominate(ins1, ins2));
    CHECK(dom.strictly_dominate(ins1, ins3));
    CHECK(dom.strictly_dominate(ins1, ins4));
    CHECK(dom.strictly_dominate(ins1, ins5));
    CHECK(dom.strictly_dominate(ins1, ins6));
    // ins2
    CHECK(dom.strictly_dominate(ins2, ins3));
    CHECK(dom.strictly_dominate(ins2, ins4));
    CHECK(dom.strictly_dominate(ins2, ins5));
    CHECK(dom.strictly_dominate(ins2, ins6));
    // ins3
    CHECK(dom.strictly_dominate(ins3, ins4));
    // ins4
    CHECK(dom.strictly_dominate(ins4, ins6));

    CHECK(not dom.strictly_dominate(ins5, ins6));
}

// clang-format off
// ┌──────────┐            
// │ins1      │            
// └┬────────┬┘            
// ┌▽──────┐┌▽─────────┐   
// │ins3   ││ins2      │   
// └┬─────┬┘└───┬─────┬┘   
// ┌▽───┐┌▽───┐┌▽───┐┌▽───┐
// │ins7││ins6││ins5││ins4│
// └┬───┘└┬───┘└┬───┘└┬───┘
// ┌▽─────▽┐┌───▽─────▽┐   
// │ins9   ││ins8      │   
// └┬──────┘└┬─────────┘   
// ┌▽────────▽┐            
// │ins10     │            
// └──────────┘
// clang-format on
TEST_CASE(dom3)
{
    migraphx::module mm;
    auto ins1 =
        mm.add_parameter("entry", {migraphx::shape::float_type}); // ins1 -> ins2, ins1 -> ins3
    auto ins2  = mm.add_instruction(pass_op{}, ins1);             // ins2 -> ins4, ins2 -> ins5
    auto ins3  = mm.add_instruction(pass_op{}, ins1);             // ins3 -> ins6, ins3 -> ins7
    auto ins4  = mm.add_instruction(pass_op{}, ins2);             // ins4 -> ins8
    auto ins5  = mm.add_instruction(pass_op{}, ins2);             // ins5 -> ins8
    auto ins6  = mm.add_instruction(pass_op{}, ins3);             // ins6 -> ins9
    auto ins7  = mm.add_instruction(pass_op{}, ins3);             // ins7 -> ins9
    auto ins8  = mm.add_instruction(pass_op{}, ins4, ins5);       // ins8 -> ins10
    auto ins9  = mm.add_instruction(pass_op{}, ins6, ins7);       // ins9 -> ins10
    auto ins10 = mm.add_instruction(pass_op{}, ins8, ins9);

    auto dom = migraphx::compute_dominator(mm);
    CHECK(not strictly_dominates_self(dom, mm));
    // ins1
    CHECK(dom.strictly_dominate(ins1, ins2));
    CHECK(dom.strictly_dominate(ins1, ins3));
    CHECK(dom.strictly_dominate(ins1, ins4));
    CHECK(dom.strictly_dominate(ins1, ins5));
    CHECK(dom.strictly_dominate(ins1, ins6));
    CHECK(dom.strictly_dominate(ins1, ins7));
    CHECK(dom.strictly_dominate(ins1, ins8));
    CHECK(dom.strictly_dominate(ins1, ins9));
    CHECK(dom.strictly_dominate(ins1, ins10));
    // ins2
    CHECK(dom.strictly_dominate(ins2, ins4));
    CHECK(dom.strictly_dominate(ins2, ins5));
    CHECK(dom.strictly_dominate(ins2, ins8));
    // ins3
    CHECK(dom.strictly_dominate(ins3, ins6));
    CHECK(dom.strictly_dominate(ins3, ins7));
    CHECK(dom.strictly_dominate(ins3, ins9));
    // ins4
    CHECK(not dom.strictly_dominate(ins4, ins8));
    CHECK(not dom.strictly_dominate(ins4, ins10));
    // ins5
    CHECK(not dom.strictly_dominate(ins5, ins8));
    CHECK(not dom.strictly_dominate(ins5, ins10));
    // ins6
    CHECK(not dom.strictly_dominate(ins6, ins9));
    CHECK(not dom.strictly_dominate(ins6, ins10));
    // ins7
    CHECK(not dom.strictly_dominate(ins7, ins9));
    CHECK(not dom.strictly_dominate(ins7, ins10));
    // ins8
    CHECK(not dom.strictly_dominate(ins8, ins10));
    // ins9
    CHECK(not dom.strictly_dominate(ins9, ins10));
}

// clang-format off
// ┌────┐    
// │ins1│    
// └┬───┘    
// ┌▽───┐    
// │ins2│    
// └┬─┬─┘    
//  │┌▽───┐  
//  ││ins3│  
//  │└┬─┬─┘  
//  │ │┌▽───┐
//  │ ││ins4│
//  │ │└┬───┘
//  │┌▽─▽─┐  
//  ││ins5│  
//  │└┬───┘  
// ┌▽─▽─┐    
// │ins6│    
// └┬───┘    
// ┌▽───┐    
// │ins7│    
// └────┘
// clang-format on
TEST_CASE(dom_unet1)
{
    migraphx::module mm;
    auto ins1 = mm.add_parameter("entry", {migraphx::shape::float_type}); // ins1 -> ins2
    auto ins2 = mm.add_instruction(pass_op{}, ins1);       // ins2 -> ins3, ins2 -> ins6
    auto ins3 = mm.add_instruction(pass_op{}, ins2);       // ins3 -> ins4, ins3 -> ins5
    auto ins4 = mm.add_instruction(pass_op{}, ins3);       // ins4 -> ins5
    auto ins5 = mm.add_instruction(pass_op{}, ins4, ins3); // ins5 -> ins6
    auto ins6 = mm.add_instruction(pass_op{}, ins5, ins2); // ins6 -> ins7
    auto ins7 = mm.add_instruction(pass_op{}, ins6);       // ins6 -> ins7

    auto dom = migraphx::compute_dominator(mm);
    CHECK(not strictly_dominates_self(dom, mm));
    // ins1
    CHECK(dom.strictly_dominate(ins1, ins2));
    CHECK(dom.strictly_dominate(ins1, ins3));
    CHECK(dom.strictly_dominate(ins1, ins4));
    CHECK(dom.strictly_dominate(ins1, ins5));
    CHECK(dom.strictly_dominate(ins1, ins6));
    CHECK(dom.strictly_dominate(ins1, ins7));
    // ins2
    CHECK(dom.strictly_dominate(ins2, ins3));
    CHECK(dom.strictly_dominate(ins2, ins4));
    CHECK(dom.strictly_dominate(ins2, ins5));
    CHECK(dom.strictly_dominate(ins2, ins6));
    CHECK(dom.strictly_dominate(ins2, ins7));
    // ins3
    CHECK(dom.strictly_dominate(ins3, ins4));
    CHECK(dom.strictly_dominate(ins3, ins5));
    // ins4
    CHECK(not dom.strictly_dominate(ins4, ins6));
    // ins5
    CHECK(not dom.strictly_dominate(ins5, ins6));
    // ins6
    CHECK(dom.strictly_dominate(ins6, ins7));
}

// clang-format off
// ┌────┐      
// │ins1│      
// └┬───┘      
// ┌▽───┐      
// │ins2│      
// └┬─┬─┘      
//  │┌▽───┐    
//  ││ins3│    
//  │└┬─┬─┘    
//  │ │┌▽───┐  
//  │ ││ins4│  
//  │ │└┬─┬─┘  
//  │ │ │┌▽───┐
//  │ │ ││ins5│
//  │ │ │└┬───┘
//  │ │┌▽─▽──┐ 
//  │ ││ins6 │ 
//  │ │└┬────┘ 
//  │┌▽─▽─┐    
//  ││ins7│    
//  │└┬───┘    
// ┌▽─▽─┐      
// │ins8│      
// └┬───┘      
// ┌▽───┐      
// │ins9│      
// └────┘
// clang-format on
TEST_CASE(dom_unet2)
{
    migraphx::module mm;
    auto ins1 = mm.add_parameter("entry", {migraphx::shape::float_type}); // ins1 -> ins2
    auto ins2 = mm.add_instruction(pass_op{}, ins1);       // ins2 -> ins3, ins2 -> ins8
    auto ins3 = mm.add_instruction(pass_op{}, ins2);       // ins3 -> ins4, ins3 -> ins7
    auto ins4 = mm.add_instruction(pass_op{}, ins3);       // ins4 -> ins5, ins4 -> ins6
    auto ins5 = mm.add_instruction(pass_op{}, ins4);       // ins6 -> ins6
    auto ins6 = mm.add_instruction(pass_op{}, ins5, ins4); // ins6 -> ins7
    auto ins7 = mm.add_instruction(pass_op{}, ins6, ins3); // ins7 -> ins8
    auto ins8 = mm.add_instruction(pass_op{}, ins7, ins2); // ins7 -> ins8
    auto ins9 = mm.add_instruction(pass_op{}, ins8);       // ins8 -> ins9

    auto dom = migraphx::compute_dominator(mm);
    CHECK(not strictly_dominates_self(dom, mm));
    // ins1
    CHECK(dom.strictly_dominate(ins1, ins2));
    CHECK(dom.strictly_dominate(ins1, ins3));
    CHECK(dom.strictly_dominate(ins1, ins4));
    CHECK(dom.strictly_dominate(ins1, ins5));
    CHECK(dom.strictly_dominate(ins1, ins6));
    CHECK(dom.strictly_dominate(ins1, ins7));
    CHECK(dom.strictly_dominate(ins1, ins8));
    CHECK(dom.strictly_dominate(ins1, ins9));
    // ins2
    CHECK(dom.strictly_dominate(ins2, ins3));
    CHECK(dom.strictly_dominate(ins2, ins4));
    CHECK(dom.strictly_dominate(ins2, ins5));
    CHECK(dom.strictly_dominate(ins2, ins6));
    CHECK(dom.strictly_dominate(ins2, ins7));
    CHECK(dom.strictly_dominate(ins2, ins8));
    CHECK(dom.strictly_dominate(ins2, ins9));
    // ins3
    CHECK(dom.strictly_dominate(ins3, ins4));
    CHECK(dom.strictly_dominate(ins3, ins5));
    CHECK(dom.strictly_dominate(ins3, ins6));
    CHECK(dom.strictly_dominate(ins3, ins7));
    // ins4
    CHECK(dom.strictly_dominate(ins4, ins5));
    CHECK(dom.strictly_dominate(ins4, ins6));
    // ins5
    CHECK(not dom.strictly_dominate(ins5, ins6));
    CHECK(not dom.strictly_dominate(ins5, ins7));
    CHECK(not dom.strictly_dominate(ins5, ins8));
    // ins6
    CHECK(not dom.strictly_dominate(ins6, ins7));
    CHECK(not dom.strictly_dominate(ins6, ins8));
    CHECK(not dom.strictly_dominate(ins6, ins9));
    // ins7
    CHECK(not dom.strictly_dominate(ins7, ins8));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
