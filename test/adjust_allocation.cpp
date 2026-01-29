/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/adjust_allocation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/module.hpp>
#include <test.hpp>

// Test allocation operation
struct test_allocate
{
    migraphx::shape s;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "test::allocate"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return s; }
};
MIGRAPHX_REGISTER_OP(test_allocate);

// Test copy operation
struct test_copy
{
    std::string name() const { return "test::copy"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        return inputs.at(1);
    }

    // Not context-free: takes context parameter
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        return args.at(1);
    }

    int output_alias(const std::vector<migraphx::shape>&) const { return 1; }
};
MIGRAPHX_REGISTER_OP(test_copy);

// Test allocation model
struct test_allocation_model
{
    std::string name() const { return "test::allocate"; }

    std::string copy() const { return "test::copy"; }

    migraphx::operation allocate(const migraphx::shape& s) const { return test_allocate{s}; }

    migraphx::operation preallocate(const migraphx::shape& s, const std::string&) const
    {
        return test_allocate{s};
    }

    bool needs_out_params() const { return false; }
};

// Test operator that takes an output buffer but returns a specific shape
// regardless of the output buffer size. This is used to test that adjust_allocation
// will reallocate when the shapes don't match.
struct simple_op
{
    migraphx::shape output_shape;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.output_shape, "output_shape"));
    }

    std::string name() const { return "simple_op"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        return output_shape;
    }

    // Not context-free: takes context parameter
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        return args.back();
    }

    // Output aliases the last input (the output buffer)
    int output_alias(const std::vector<migraphx::shape>& inputs) const
    {
        return static_cast<int>(inputs.size()) - 1;
    }
};

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m,
        {migraphx::adjust_allocation{test_allocation_model{}}, migraphx::dead_code_elimination{}});
}

// Test that adjust_allocation reallocates when the output shape differs from the allocated shape
TEST_CASE(realloc_shape_mismatch)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        // Allocate a buffer with shape {3, 2} but the operator returns shape {2, 3}
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {3, 2}}});
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        // After adjust_allocation, the allocate should have the correct shape {2, 3}
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        m2.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc);
    }

    EXPECT(m1 == m2);
}

// Test that adjust_allocation does nothing when shapes already match
TEST_CASE(no_realloc_shape_match)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        // Allocate a buffer with the same shape as the operator output
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that adjust_allocation skips when output alias is a parameter with matching shape
TEST_CASE(skip_output_param_shape_match)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        // Output parameter with same shape as what the operator produces
        auto out = m1.add_parameter("output", {migraphx::shape::float_type, {2, 3}});
        auto r   = m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, out);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that context-free operations are skipped
TEST_CASE(skip_context_free_op)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        // Use a context-free operation (add is context-free)
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, x);
        m1.add_return({sum});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test with non-standard strides in allocation
TEST_CASE(realloc_nonstandard_strides)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {3, 2}});
        // Allocate with transposed strides {1, 3} but operator expects standard strides
        auto alloc =
            m1.add_instruction(test_allocate{{migraphx::shape::float_type, {3, 2}, {1, 3}}});
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, x, alloc);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {3, 2}});
        // After adjust_allocation, should have standard strides
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {3, 2}}});
        m2.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, x, alloc);
    }

    EXPECT(m1 == m2);
}

// Test with different data types
TEST_CASE(realloc_different_dtype)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::half_type, {4, 4}});
        // Allocate with wrong dimensions for half type
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::half_type, {2, 8}}});
        m1.add_instruction(simple_op{{migraphx::shape::half_type, {4, 4}}}, x, alloc);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::half_type, {4, 4}});
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::half_type, {4, 4}}});
        m2.add_instruction(simple_op{{migraphx::shape::half_type, {4, 4}}}, x, alloc);
    }

    EXPECT(m1 == m2);
}

// Test with multiple instructions, only some need reallocation
TEST_CASE(realloc_mixed_instructions)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto y = m1.add_parameter("y", {migraphx::shape::float_type, {3, 2}});

        // First op: needs reallocation (allocated {3, 2} but returns {2, 3})
        auto alloc1 = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {3, 2}}});
        auto r1 = m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc1);

        // Second op: no reallocation needed (shapes match)
        auto alloc2 = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {3, 2}}});
        auto r2 = m1.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, y, alloc2);

        auto sum = m1.add_instruction(migraphx::make_op("add"), r1, r1);
        m1.add_return({sum, r2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto y = m2.add_parameter("y", {migraphx::shape::float_type, {3, 2}});

        // First op: reallocated to correct shape
        auto alloc1 = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        auto r1 = m2.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc1);

        // Second op: unchanged
        auto alloc2 = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {3, 2}}});
        auto r2 = m2.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, y, alloc2);

        auto sum = m2.add_instruction(migraphx::make_op("add"), r1, r1);
        m2.add_return({sum, r2});
    }

    EXPECT(m1 == m2);
}

// Test that instructions with no inputs are skipped
TEST_CASE(skip_no_inputs)
{
    migraphx::module m1;
    {
        auto lit =
            m1.add_literal(migraphx::literal{{migraphx::shape::float_type, {2, 2}}, {1, 2, 3, 4}});
        m1.add_return({lit});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test 3D tensor reallocation
TEST_CASE(realloc_3d_tensor)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3, 4}});
        // Allocate wrong 3D shape
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {4, 3, 2}}});
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3, 4}}}, x, alloc);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3, 4}});
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3, 4}}});
        m2.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3, 4}}}, x, alloc);
    }

    EXPECT(m1 == m2);
}

// Test that adjust_allocation inserts a copy when output alias is a parameter with different shape
TEST_CASE(insert_copy_output_param)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        // Output parameter with different shape than what the operator produces
        auto out = m1.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        auto r   = m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, out);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto out = m2.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        // New allocation with correct shape replaces the parameter
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        auto r     = m2.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc);
        // Copy from result to output parameter
        auto c = m2.add_instruction(migraphx::make_op("test::copy"), r, out);
        m2.add_return({c});
    }

    EXPECT(m1 == m2);
}

// Test that copy insertion updates multiple users of the result
TEST_CASE(insert_copy_multiple_users)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto out = m1.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        auto r   = m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, out);
        // Multiple uses of the result after the instruction
        auto sum = m1.add_instruction(migraphx::make_op("add"), r, r);
        m1.add_return({sum});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto out   = m2.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        auto r     = m2.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc);
        // Copy from result to output parameter
        auto c = m2.add_instruction(migraphx::make_op("test::copy"), r, out);
        // Users after the copy should use the copy result
        auto sum = m2.add_instruction(migraphx::make_op("add"), c, c);
        m2.add_return({sum});
    }

    EXPECT(m1 == m2);
}

// Test copy insertion with chain of operations using the result
TEST_CASE(insert_copy_chain_of_ops)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto out = m1.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        auto r   = m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, out);
        // Chain of operations using r
        auto neg  = m1.add_instruction(migraphx::make_op("neg"), r);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), neg);
        m1.add_return({relu});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x     = m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto out   = m2.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        auto r     = m2.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, alloc);
        auto c     = m2.add_instruction(migraphx::make_op("test::copy"), r, out);
        // Chain uses the copy result
        auto neg  = m2.add_instruction(migraphx::make_op("neg"), c);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), neg);
        m2.add_return({relu});
    }

    EXPECT(m1 == m2);
}

// Test copy insertion with non-standard strides in output param
TEST_CASE(insert_copy_nonstandard_strides)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {3, 2}});
        // Output parameter with transposed strides
        auto out = m1.add_parameter("output", {migraphx::shape::float_type, {3, 2}, {1, 3}});
        auto r   = m1.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, x, out);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", {migraphx::shape::float_type, {3, 2}});
        auto out = m2.add_parameter("output", {migraphx::shape::float_type, {3, 2}, {1, 3}});
        // Allocate with standard strides
        auto alloc = m2.add_instruction(test_allocate{{migraphx::shape::float_type, {3, 2}}});
        auto r     = m2.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, x, alloc);
        auto c     = m2.add_instruction(migraphx::make_op("test::copy"), r, out);
        m2.add_return({c});
    }

    EXPECT(m1 == m2);
}

// Test that copy is not inserted when result is not used after the instruction
TEST_CASE(insert_copy_result_not_used_later)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto y   = m1.add_parameter("y", {migraphx::shape::float_type, {2, 3}});
        auto out = m1.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        // r is not used after itself - DCE will remove this and the copy
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, out);
        // y is used in return, not r
        m1.add_return({y});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        m2.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto y = m2.add_parameter("y", {migraphx::shape::float_type, {2, 3}});
        m2.add_parameter("output", {migraphx::shape::float_type, {3, 2}});
        // After DCE, the simple_op, allocate and copy are all removed
        m2.add_return({y});
    }

    EXPECT(m1 == m2);
}

// Test that view operations as output buffer are not modified (shallow alias only traces one level)
// The pass uses shallow=true, so it only looks at the immediate output alias, not through view ops
TEST_CASE(skip_aliased_through_transpose)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {3, 2}});
        // Allocate {2, 3} and transpose to {3, 2} - pass won't modify since alias is transpose, not
        // allocate
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        auto t =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), alloc);
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, x, t);
    }
    // With shallow=true aliasing, the pass sees transpose as the alias, not the allocate
    // Since transpose is not an allocate or parameter, the pass skips this instruction
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that squeeze as output buffer is skipped (shallow alias)
TEST_CASE(skip_aliased_through_squeeze)
{
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {1, 2, 3}}});
        auto sq    = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), alloc);
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, sq);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that unsqueeze as output buffer is skipped (shallow alias)
TEST_CASE(skip_aliased_through_unsqueeze)
{
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2, 3}});
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {2, 3}}});
        auto usq   = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), alloc);
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {1, 2, 3}}}, x, usq);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that slice as output buffer is skipped (shallow alias)
TEST_CASE(skip_aliased_through_slice)
{
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
        auto alloc = m1.add_instruction(test_allocate{{migraphx::shape::float_type, {4, 6}}});
        auto sl    = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {2, 3}}}),
            alloc);
        m1.add_instruction(simple_op{{migraphx::shape::float_type, {2, 3}}}, x, sl);
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that transposed parameter as output buffer is skipped (shallow alias)
TEST_CASE(skip_aliased_param_through_transpose)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {3, 2}});
        auto out = m1.add_parameter("output", {migraphx::shape::float_type, {2, 3}});
        auto t = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), out);
        auto r = m1.add_instruction(simple_op{{migraphx::shape::float_type, {3, 2}}}, x, t);
        m1.add_return({r});
    }
    // Shallow alias sees transpose, not the parameter, so pass skips this
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
