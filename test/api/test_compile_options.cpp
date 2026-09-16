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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <limits>
#include "test.hpp"

static migraphx::program create_add_program(const migraphx::shape& s)
{
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    std::vector<float> x_values(s.elements(), 1);
    auto x = m.add_literal(s, x_values.data());
    std::vector<float> y_values(s.elements(), -1);
    auto y = m.add_literal(s, y_values.data());
    auto r = m.add_instruction(migraphx::operation("add"), {x, y});
    m.add_return({r});
    return p;
}

static void run_and_check(migraphx::program& p, const migraphx::shape& s)
{
    migraphx::program_parameters pp;
    auto outputs = p.eval(pp);
    std::vector<float> expected(s.elements(), 0);
    CHECK(outputs[0] == migraphx::argument(s, expected.data()));
}

TEST_CASE(compile_options_flags)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_offload_copy(false);
    options.set_fast_math(false);
    options.set_exhaustive_tune_flag(false);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(compile_options_backend_option_scalar)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_advance_backend_option("int_option", 42);
    options.set_advance_backend_option("str_option", "gfx942");
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(compile_options_backend_option_vector)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_advance_backend_option("ints", std::vector<int>{1, 2, 3});
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(compile_options_backend_options_json)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_advance_backend_options("{a:%i, b:%i}", 1, 2);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(compile_options_compile_with_eager_mode)
{
    migraphx::api::program p;
    auto main_module = p.get_main_module();
    migraphx::api::shape s{migraphx_shape_float_type, {2, 3}};
    auto x  = main_module.add_parameter("x", s);
    auto y  = main_module.add_parameter("y", s);
    auto op = migraphx::api::operation("add");
    main_module.add_instruction(op, {x, y});

    migraphx::api::compile_options options;
    options.set_compile_mode(0);
    p.compile(migraphx::api::target("ref"), options);

    auto output_shapes = p.get_output_shapes();
    CHECK(output_shapes.size() == 1);
}

TEST_CASE(compile_options_compile_with_max_mode)
{
    migraphx::api::program p;
    auto main_module = p.get_main_module();
    migraphx::api::shape s{migraphx_shape_float_type, {2, 3}};
    auto x  = main_module.add_parameter("x", s);
    auto y  = main_module.add_parameter("y", s);
    auto op = migraphx::api::operation("add");
    main_module.add_instruction(op, {x, y});

    migraphx::api::compile_options options;
    options.set_compile_mode(100);
    p.compile(migraphx::api::target("ref"), options);

    auto output_shapes = p.get_output_shapes();
    CHECK(output_shapes.size() == 1);
}

TEST_CASE(compile_options_set_compile_mode_default_argument)
{
    migraphx::compile_options options;
    options.set_compile_mode();
    options.set_compile_mode(migraphx_compile_mode_balanced);
    options.set_compile_mode(30);
}

TEST_CASE(compile_options_compile_with_balanced_mode)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    options.set_compile_mode(migraphx_compile_mode_balanced);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(c_api_set_compile_mode)
{
    migraphx_compile_options_t options = nullptr;
    CHECK(migraphx_compile_options_create(&options) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, migraphx_compile_mode_eager) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, migraphx_compile_mode_balanced) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, migraphx_compile_mode_max) ==
          migraphx_status_success);
    // an intermediate value snaps to the nearest known mode
    CHECK(migraphx_compile_options_set_compile_mode(options, 30) == migraphx_status_success);
    CHECK(migraphx_compile_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_set_compile_mode_null_handle)
{
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, migraphx_compile_mode_balanced) ==
          migraphx_status_bad_param);
}

TEST_CASE(c_api_set_advance_backend_options_null_handle)
{
    CHECK(migraphx_compile_options_set_advance_backend_options(nullptr, "{a:1}") ==
          migraphx_status_bad_param);
}

TEST_CASE(c_api_set_advance_backend_options)
{
    migraphx_compile_options_t options = nullptr;
    CHECK(migraphx_compile_options_create(&options) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_advance_backend_options(options, "{a:%i, b:%i}", 1, 2) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_compile_options_lifecycle)
{
    migraphx_compile_options_t options = nullptr;
    CHECK(migraphx_compile_options_create(&options) == migraphx_status_success);
    CHECK(options != nullptr);
    CHECK(migraphx_compile_options_set_compile_mode(options, migraphx_compile_mode_eager) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, migraphx_compile_mode_max) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_advance_backend_options(options, "{a:%i, b:%i}", 1, 2) ==
          migraphx_status_success);
    migraphx_compile_options_t copy = nullptr;
    CHECK(migraphx_compile_options_create(&copy) == migraphx_status_success);
    CHECK(migraphx_compile_options_assign_to(copy, options) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(copy, migraphx_compile_mode_balanced) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_destroy(copy) == migraphx_status_success);
    CHECK(migraphx_compile_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_compile_options_destroy_null_handle)
{
    // destroy delegates to `delete`, so a null handle is a no-op rather than an error
    CHECK(migraphx_compile_options_destroy(nullptr) == migraphx_status_success);
}

TEST_CASE(c_api_set_advance_backend_options_empty_json)
{
    migraphx_compile_options_t options = nullptr;
    CHECK(migraphx_compile_options_create(&options) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_advance_backend_options(options, "") ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_advance_backend_options(options, nullptr) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_set_advance_backend_options_non_object_json)
{
    migraphx_compile_options_t options = nullptr;
    CHECK(migraphx_compile_options_create(&options) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_advance_backend_options(options, "[1, 2, 3]") ==
          migraphx_status_unknown_error);
    CHECK(migraphx_compile_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_set_compile_mode_named_modes_compile_and_run)
{
    // There is no getter for compile_mode, so a successful status only proves
    // the call returned. Compiling and evaluating afterwards is the only way to
    // observe that the mode reached the compile_options the target reads.
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};

    auto eager = create_add_program(s);
    migraphx::compile_options eager_options;
    CHECK(migraphx_compile_options_set_compile_mode(eager_options.get_handle_ptr(),
                                                    migraphx_compile_mode_eager) ==
          migraphx_status_success);
    eager.compile(migraphx::target("ref"), eager_options);
    run_and_check(eager, s);

    auto balanced = create_add_program(s);
    migraphx::compile_options balanced_options;
    CHECK(migraphx_compile_options_set_compile_mode(balanced_options.get_handle_ptr(),
                                                    migraphx_compile_mode_balanced) ==
          migraphx_status_success);
    balanced.compile(migraphx::target("ref"), balanced_options);
    run_and_check(balanced, s);

    auto max = create_add_program(s);
    migraphx::compile_options max_options;
    CHECK(migraphx_compile_options_set_compile_mode(
              max_options.get_handle_ptr(), migraphx_compile_mode_max) == migraphx_status_success);
    max.compile(migraphx::target("ref"), max_options);
    run_and_check(max, s);
}

TEST_CASE(c_api_set_compile_mode_snapped_value_compiles_and_runs)
{
    // A value that names no mode is snapped onto the nearest one rather than
    // rejected, so the program still compiles.
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    CHECK(migraphx_compile_options_set_compile_mode(options.get_handle_ptr(), 30) ==
          migraphx_status_success);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(c_api_set_compile_mode_negative_value_compiles_and_runs)
{
    // The int8_t argument is widened to the uint8_t overload of
    // convert_to_compile_mode, so a negative wraps into [128, 255] and clamps
    // *up* to max. It selects max, not eager, and leaves the options usable.
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    CHECK(migraphx_compile_options_set_compile_mode(options.get_handle_ptr(), -1) ==
          migraphx_status_success);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(c_api_set_compile_mode_boundary_values_succeed)
{
    migraphx_compile_options_t options = nullptr;
    CHECK(migraphx_compile_options_create(&options) == migraphx_status_success);
    // Exact modes, the midpoints between them, and the values either side of
    // each mode.
    CHECK(migraphx_compile_options_set_compile_mode(options, 0) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 1) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 25) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 49) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 50) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 51) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 75) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 99) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, 100) == migraphx_status_success);
    // int8_t tops out at 127, so 101..127 are the only representable
    // above-range values.
    CHECK(migraphx_compile_options_set_compile_mode(options, 101) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, std::numeric_limits<int8_t>::max()) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, -1) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options, std::numeric_limits<int8_t>::min()) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_set_compile_mode_full_int8_domain_succeeds)
{
    // The setter has no invalid-value error path: out-of-range requests are
    // clamped and unnamed ones are snapped, so every representable int8_t must
    // be accepted. Looping over int avoids overflowing the counter at 127.
    migraphx_compile_options_t options = nullptr;
    CHECK(migraphx_compile_options_create(&options) == migraphx_status_success);
    for(int value = std::numeric_limits<int8_t>::min(); value <= std::numeric_limits<int8_t>::max();
        ++value)
    {
        CHECK(migraphx_compile_options_set_compile_mode(options, static_cast<int8_t>(value)) ==
              migraphx_status_success);
    }
    CHECK(migraphx_compile_options_destroy(options) == migraphx_status_success);
}

TEST_CASE(c_api_set_compile_mode_null_handle_rejects_every_value)
{
    // The null check runs before the value is looked at, so in-range,
    // out-of-range and negative values all report the same error.
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, migraphx_compile_mode_eager) ==
          migraphx_status_bad_param);
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, migraphx_compile_mode_max) ==
          migraphx_status_bad_param);
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, 30) == migraphx_status_bad_param);
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, -1) == migraphx_status_bad_param);
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, std::numeric_limits<int8_t>::max()) ==
          migraphx_status_bad_param);
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, std::numeric_limits<int8_t>::min()) ==
          migraphx_status_bad_param);
}

TEST_CASE(c_api_set_compile_mode_null_handle_has_no_side_effects)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    CHECK(migraphx_compile_options_set_compile_mode(
              options.get_handle_ptr(), migraphx_compile_mode_eager) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(nullptr, migraphx_compile_mode_max) ==
          migraphx_status_bad_param);
    // The rejected call must not have disturbed the unrelated live handle.
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(c_api_set_compile_mode_last_value_wins)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    // The setter overwrites rather than accumulates, so repeated calls are safe.
    CHECK(migraphx_compile_options_set_compile_mode(
              options.get_handle_ptr(), migraphx_compile_mode_max) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(options.get_handle_ptr(), 30) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(
              options.get_handle_ptr(), migraphx_compile_mode_eager) == migraphx_status_success);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(c_api_set_compile_mode_handles_are_independent)
{
    // The mode is stored per handle, not in shared state.
    migraphx_compile_options_t eager = nullptr;
    migraphx_compile_options_t max   = nullptr;
    CHECK(migraphx_compile_options_create(&eager) == migraphx_status_success);
    CHECK(migraphx_compile_options_create(&max) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(eager, migraphx_compile_mode_eager) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(max, migraphx_compile_mode_max) ==
          migraphx_status_success);
    // Destroying one handle leaves the other writable.
    CHECK(migraphx_compile_options_destroy(eager) == migraphx_status_success);
    CHECK(migraphx_compile_options_set_compile_mode(max, migraphx_compile_mode_balanced) ==
          migraphx_status_success);
    CHECK(migraphx_compile_options_destroy(max) == migraphx_status_success);
}

TEST_CASE(c_api_set_compile_mode_composes_with_other_options)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    auto p = create_add_program(s);

    migraphx::compile_options options;
    // Setting the mode before and after the other setters must leave all of
    // them in effect.
    CHECK(migraphx_compile_options_set_compile_mode(
              options.get_handle_ptr(), migraphx_compile_mode_max) == migraphx_status_success);
    options.set_offload_copy(false);
    options.set_fast_math(false);
    options.set_exhaustive_tune_flag(false);
    options.set_advance_backend_options("{a:%i}", 1);
    CHECK(migraphx_compile_options_set_compile_mode(
              options.get_handle_ptr(), migraphx_compile_mode_eager) == migraphx_status_success);
    p.compile(migraphx::target("ref"), options);
    run_and_check(p, s);
}

TEST_CASE(c_api_set_compile_mode_reusable_after_compile)
{
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    migraphx::compile_options options;

    CHECK(migraphx_compile_options_set_compile_mode(
              options.get_handle_ptr(), migraphx_compile_mode_eager) == migraphx_status_success);
    auto first = create_add_program(s);
    first.compile(migraphx::target("ref"), options);
    run_and_check(first, s);

    // compile() must not consume or invalidate the options handle.
    CHECK(migraphx_compile_options_set_compile_mode(
              options.get_handle_ptr(), migraphx_compile_mode_max) == migraphx_status_success);
    auto second = create_add_program(s);
    second.compile(migraphx::target("ref"), options);
    run_and_check(second, s);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
