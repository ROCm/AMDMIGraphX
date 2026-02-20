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
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/load.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct test_concat : migraphx::auto_register_op<test_concat>
{
    test_concat() = default;
    test_concat(std::size_t axis) { op.axis = axis; }
    migraphx::op::concat op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    migraphx::value attributes() const
    {
        migraphx::value normalize;
        normalize["axis"] = migraphx::value::array{migraphx::op::normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "test::concat"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        auto out = inputs.back();
        inputs.pop_back();
        auto result = op.normalize_compute_shape(std::move(inputs));
        if(result != out)
            MIGRAPHX_THROW("Allocation doesn't match");
        return result;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

struct test_copy : migraphx::auto_register_op<test_copy>
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return migraphx::pack();
    }

    std::string name() const { return "test::copy"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(2).same_dims();
        return inputs.at(1);
    }
    migraphx::argument compute(const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>& args) const
    {
        migraphx::argument result{output_shape};

        visit_all(result, args[0])([&](auto output, auto input) {
            std::copy(input.begin(), input.end(), output.begin());
        });

        return result;
    }

    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};

struct test_allocate : migraphx::auto_register_op<test_allocate>
{
    migraphx::shape s{};

    test_allocate() = default;
    test_allocate(migraphx::shape ss) : migraphx::auto_register_op<test_allocate>(), s(ss) {}

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "test::allocate"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

struct test_allocation_model
{
    std::string name() const { return "test::allocate"; }
    std::string copy() const { return "test::copy"; }
    migraphx::operation allocate(const migraphx::shape& s) const
    {
        return migraphx::make_op(name(), {{"shape", to_value(s)}});
    }
    migraphx::operation preallocate(const migraphx::shape&, const std::string&) const
    {
        MIGRAPHX_THROW("preallocate is not used by eliminate_concat");
    }
    bool needs_out_params() const { return false; }
};

struct concat_test_optimization
{
    std::unordered_set<std::string> op_non_packed_output = {};
    /// A unique name used to identify the allocate operator
    std::string allocate() const { return "allocate"; }
    /// Return the lowered concat operator
    std::optional<migraphx::op::concat> get_concat(const migraphx::operation& op) const
    {
        if(op.name() != "test::concat")
            return std::nullopt;
        return migraphx::any_cast<test_concat>(op).op;
    }

    bool supports_non_packed_output(migraphx::instruction_ref ins) const
    {
        if(migraphx::contains(op_non_packed_output, "*"))
            return true;
        return migraphx::contains(op_non_packed_output, ins->name());
    }

    test_allocation_model allocation() const { return test_allocation_model{}; }
};

static void run_pass(migraphx::module& m, const concat_test_optimization& opt = {})
{
    migraphx::run_passes(m, {migraphx::eliminate_concat{opt}, migraphx::dead_code_elimination{}});
}

struct simple_op
{
    std::string name() const { return "simple_op"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(1);
        return inputs.at(0);
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>& args) const
    {
        return args.at(0);
    }
    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>&) const { return {0}; }
};

template <std::size_t... Is, class... Ts>
static migraphx::shape create_shape(Ts... xs)
{
    if(sizeof...(Is) == 0)
        return migraphx::shape{migraphx::shape::float_type, {std::size_t(xs)...}};
    else
        return migraphx::shape::from_permutation(
            migraphx::shape::float_type, {std::size_t(xs)...}, {Is...});
}

template <std::size_t... Is, class... Ts>
static migraphx::operation make_allocate(Ts... xs)
{
    return migraphx::make_op("test::allocate", {{"shape", to_value(create_shape<Is...>(xs...))}});
}

using load     = migraphx::op::load;
using identity = migraphx::op::identity;

TEST_CASE(simple)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(1));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(1));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(2));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 0}}), s1, s2, a3);
        m1.add_return({c1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(2));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), a1);
        auto s1     = m2.add_instruction(simple_op{}, slice1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), a1);
        auto s2  = m2.add_instruction(simple_op{}, slice2);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, s1, s2);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(negative_axis_last_axis)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(2, 2));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(2, 2));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(2, 4));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", -1}}), s1, s2, a3);
        m1.add_return({c1});
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(negative_axis_last_axis_support_non_packed)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(2, 2));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(2, 2));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(2, 4));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", -1}}), s1, s2, a3);
        m1.add_return({c1});
    }
    run_pass(m1, {.op_non_packed_output = {"simple_op"}});
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(2, 4));
        auto a2     = m2.add_instruction(make_allocate(2, 2));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), a1);
        auto s1     = m2.add_instruction(simple_op{}, a2);
        auto cp1    = m2.add_instruction(migraphx::make_op("test::copy"), s1, slice1);
        auto a3     = m2.add_instruction(make_allocate(2, 2));
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {4}}}), a1);
        auto s2  = m2.add_instruction(simple_op{}, a3);
        auto cp2 = m2.add_instruction(migraphx::make_op("test::copy"), s2, slice2);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, cp1, cp2);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(negative_axis_first_axis)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(2, 2));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(2, 2));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(4, 2));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", -2}}), s1, s2, a3);
        m1.add_return({c1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(4, 2));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), a1);
        auto s1     = m2.add_instruction(simple_op{}, slice1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), a1);
        auto s2  = m2.add_instruction(simple_op{}, slice2);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, s1, s2);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(negative_axis_middle_axis_with_empty_axis0)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(1, 2, 2));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(1, 2, 2));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(1, 4, 2));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", -2}}), s1, s2, a3);
        m1.add_return({c1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(1, 4, 2));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), a1);
        auto s1     = m2.add_instruction(simple_op{}, slice1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {4}}}), a1);
        auto s2  = m2.add_instruction(simple_op{}, slice2);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, s1, s2);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reversed_arguments)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(1));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(1));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(2));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 0}}), s2, s1, a3);
        m1.add_return({c1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(2));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), a1);
        auto s1     = m2.add_instruction(simple_op{}, slice1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), a1);
        auto s2  = m2.add_instruction(simple_op{}, slice2);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, s2, s1);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nested)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(1));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(1));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(2));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 0}}), s1, s2, a3);
        auto a4 = m1.add_instruction(make_allocate(1));
        auto s3 = m1.add_instruction(simple_op{}, a4);
        auto a5 = m1.add_instruction(make_allocate(1));
        auto s4 = m1.add_instruction(simple_op{}, a5);
        auto a6 = m1.add_instruction(make_allocate(2));
        auto c2 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 0}}), s3, s4, a6);
        auto a7 = m1.add_instruction(make_allocate(4));
        auto c3 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 0}}), c1, c2, a7);
        m1.add_return({c3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(4));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), a1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), a1);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), slice2);
        auto slice4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), slice2);
        auto s1     = m2.add_instruction(simple_op{}, slice4);
        auto s2     = m2.add_instruction(simple_op{}, slice3);
        auto id1    = m2.add_instruction(migraphx::make_op("identity"), slice2, s1, s2);
        auto slice5 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), slice1);
        auto slice6 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), slice1);
        auto s3  = m2.add_instruction(simple_op{}, slice6);
        auto s4  = m2.add_instruction(simple_op{}, slice5);
        auto id2 = m2.add_instruction(migraphx::make_op("identity"), slice1, s3, s4);
        auto id3 = m2.add_instruction(migraphx::make_op("identity"), a1, id1, id2);
        m2.add_return({id3});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(concat_axis1_with_empty_axis0)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(1, 2, 8, 8));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(1, 3, 8, 8));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(1, 5, 8, 8));
        auto s3 = m1.add_instruction(simple_op{}, a3);
        auto a4 = m1.add_instruction(make_allocate(1, 10, 8, 8));
        auto c1 =
            m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 1}}), s1, s2, s3, a4);
        m1.add_return({c1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(1, 10, 8, 8));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), a1);
        auto s1     = m2.add_instruction(simple_op{}, slice1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {5}}}), a1);
        auto s2     = m2.add_instruction(simple_op{}, slice2);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {5}}, {"ends", {10}}}), a1);
        auto s3  = m2.add_instruction(simple_op{}, slice3);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, s1, s2, s3);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(concat_axis1_with_empty_axis0_nwhc)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 2, 8, 8));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 3, 8, 8));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 5, 8, 8));
        auto s3 = m1.add_instruction(simple_op{}, a3);
        auto a4 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 10, 8, 8));
        auto c1 =
            m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 1}}), s1, s2, s3, a4);
        m1.add_return({c1});
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(concat_axis1_with_empty_axis0_nhwc_supports_non_packed_output)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 2, 8, 8));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 3, 8, 8));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 5, 8, 8));
        auto s3 = m1.add_instruction(simple_op{}, a3);
        auto a4 = m1.add_instruction(make_allocate<0, 2, 3, 1>(1, 10, 8, 8));
        auto c1 =
            m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 1}}), s1, s2, s3, a4);
        m1.add_return({c1});
    }
    run_pass(m1, {.op_non_packed_output = {"simple_op"}});
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate<0, 2, 3, 1>(1, 10, 8, 8));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {5}}, {"ends", {10}}}), a1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {5}}}), a1);
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), a1);
        auto a2  = m2.add_instruction(make_allocate<0, 2, 3, 1>(1, 2, 8, 8));
        auto s1  = m2.add_instruction(simple_op{}, a2);
        auto cp1 = m2.add_instruction(migraphx::make_op("test::copy"), s1, slice3);
        auto a3  = m2.add_instruction(make_allocate<0, 2, 3, 1>(1, 3, 8, 8));
        auto s2  = m2.add_instruction(simple_op{}, a3);
        auto cp2 = m2.add_instruction(migraphx::make_op("test::copy"), s2, slice2);
        auto a4  = m2.add_instruction(make_allocate<0, 2, 3, 1>(1, 5, 8, 8));
        auto s3  = m2.add_instruction(simple_op{}, a4);
        auto cp3 = m2.add_instruction(migraphx::make_op("test::copy"), s3, slice1);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, cp1, cp2, cp3);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(non_packed_output_not_supported)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(2, 2, 8, 8));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(2, 3, 8, 8));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(2, 5, 8, 8));
        auto s3 = m1.add_instruction(simple_op{}, a3);
        auto a4 = m1.add_instruction(make_allocate(2, 10, 8, 8));
        auto c1 =
            m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 1}}), s1, s2, s3, a4);
        m1.add_return({c1});
    }
    migraphx::module m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(non_packed_output_supported)
{
    migraphx::module m1;
    {
        auto a1 = m1.add_instruction(make_allocate(2, 2, 8, 8));
        auto s1 = m1.add_instruction(simple_op{}, a1);
        auto a2 = m1.add_instruction(make_allocate(2, 3, 8, 8));
        auto s2 = m1.add_instruction(simple_op{}, a2);
        auto a3 = m1.add_instruction(make_allocate(2, 5, 8, 8));
        auto s3 = m1.add_instruction(simple_op{}, a3);
        auto a4 = m1.add_instruction(make_allocate(2, 10, 8, 8));
        auto c1 =
            m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 1}}), s1, s2, s3, a4);
        m1.add_return({c1});
    }
    run_pass(m1, {.op_non_packed_output = {"simple_op"}});
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(2, 10, 8, 8));
        auto a2     = m2.add_instruction(make_allocate(2, 2, 8, 8));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), a1);
        auto s1     = m2.add_instruction(simple_op{}, a2);
        auto cp1    = m2.add_instruction(migraphx::make_op("test::copy"), s1, slice1);
        auto a3     = m2.add_instruction(make_allocate(2, 3, 8, 8));
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {5}}}), a1);
        auto s2     = m2.add_instruction(simple_op{}, a3);
        auto cp2    = m2.add_instruction(migraphx::make_op("test::copy"), s2, slice2);
        auto a4     = m2.add_instruction(make_allocate(2, 5, 8, 8));
        auto slice3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {5}}, {"ends", {10}}}), a1);
        auto s3  = m2.add_instruction(simple_op{}, a4);
        auto cp3 = m2.add_instruction(migraphx::make_op("test::copy"), s3, slice3);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, cp1, cp2, cp3);
        m2.add_return({id1});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(one_copy_with_one_broadcasted_input)
{
    migraphx::module m1;
    {
        auto a1  = m1.add_instruction(make_allocate(4, 16, 16));
        auto s1  = m1.add_instruction(simple_op{}, a1);
        auto a2  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s2  = m1.add_instruction(simple_op{}, a2);
        auto s2b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s2);
        auto a3 = m1.add_instruction(make_allocate(6, 16, 16));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 0}}), s1, s2b, a3);
        m1.add_return({c1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(6, 16, 16));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {6}}}), a1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), a1);
        auto s1  = m2.add_instruction(simple_op{}, slice2);
        auto a2  = m2.add_instruction(make_allocate(2, 16, 1));
        auto s2  = m2.add_instruction(simple_op{}, a2);
        auto a2b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}, {"out_dyn_dims", {}}}),
            s2);
        auto cp1 = m2.add_instruction(migraphx::make_op("test::copy"), a2b, slice1);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, s1, cp1);
        m2.add_return({id1});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(one_copy_with_one_broadcasted_input_support_non_packed)
{
    migraphx::module m1;
    {
        auto a1  = m1.add_instruction(make_allocate(4, 16, 16));
        auto s1  = m1.add_instruction(simple_op{}, a1);
        auto a2  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s2  = m1.add_instruction(simple_op{}, a2);
        auto s2b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s2);
        auto a3 = m1.add_instruction(make_allocate(6, 16, 16));
        auto c1 = m1.add_instruction(migraphx::make_op("test::concat", {{"axis", 0}}), s1, s2b, a3);
        m1.add_return({c1});
    }
    run_pass(m1, {.op_non_packed_output = {"*"}});
    migraphx::module m2;
    {
        auto a1     = m2.add_instruction(make_allocate(6, 16, 16));
        auto slice1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {6}}}), a1);
        auto slice2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), a1);
        auto s1  = m2.add_instruction(simple_op{}, slice2);
        auto a2  = m2.add_instruction(make_allocate(2, 16, 1));
        auto s2  = m2.add_instruction(simple_op{}, a2);
        auto a2b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}, {"out_dyn_dims", {}}}),
            s2);
        auto cp1 = m2.add_instruction(migraphx::make_op("test::copy"), a2b, slice1);
        auto id1 = m2.add_instruction(migraphx::make_op("identity"), a1, s1, cp1);
        m2.add_return({id1});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(concat_with_three_broadcasted_inputs)
{
    migraphx::module m1;
    {
        auto a1  = m1.add_instruction(make_allocate(4, 16, 16));
        auto s1  = m1.add_instruction(simple_op{}, a1);
        auto a2  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s2  = m1.add_instruction(simple_op{}, a2);
        auto s2b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s2);
        auto a3  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s3  = m1.add_instruction(simple_op{}, a3);
        auto s3b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s3);
        auto a4  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s4  = m1.add_instruction(simple_op{}, a4);
        auto s4b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s4);
        auto a5 = m1.add_instruction(make_allocate(10, 16, 16));
        auto c1 = m1.add_instruction(
            migraphx::make_op("test::concat", {{"axis", 0}}), s1, s2b, s3b, s4b, a5);
        m1.add_return({c1});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(concat_with_three_broadcasted_inputs_support_non_packed)
{
    migraphx::module m1;
    {
        auto a1  = m1.add_instruction(make_allocate(4, 16, 16));
        auto s1  = m1.add_instruction(simple_op{}, a1);
        auto a2  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s2  = m1.add_instruction(simple_op{}, a2);
        auto s2b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s2);
        auto a3  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s3  = m1.add_instruction(simple_op{}, a3);
        auto s3b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s3);
        auto a4  = m1.add_instruction(make_allocate(2, 16, 1));
        auto s4  = m1.add_instruction(simple_op{}, a4);
        auto s4b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 16}}}), s4);
        auto a5 = m1.add_instruction(make_allocate(10, 16, 16));
        auto c1 = m1.add_instruction(
            migraphx::make_op("test::concat", {{"axis", 0}}), s1, s2b, s3b, s4b, a5);
        m1.add_return({c1});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.op_non_packed_output = {"*"}});
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
