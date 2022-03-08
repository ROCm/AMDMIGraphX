#include <numeric>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

void run_test_on_gpu(const migraphx::program p) {
    // make a copy of the program 
    migraphx_program_t p_ref_c;
    migraphx_program_create(&p_ref_c);
    migraphx_program_assign_to(p_ref_c, p.get_handle_ptr());
    auto p_ref = migraphx::program(p_ref_c, migraphx::own{});
    // compile on GPU
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    // set parameters_map
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    // get results from GPU
    auto outputs = p.eval(pp);
    auto output  = outputs[0];
    auto lens    = output.get_shape().lengths();
    auto elem_num = std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<std::size_t>());
    float* output_data_ptr = reinterpret_cast<float*>(output.data());
    std::vector<float> output_vec(output_data_ptr, output_data_ptr + elem_num);
    // compile using ref target
    p_ref.compile(migraphx::target("ref"));
    // get results from Ref target
    auto outputs_ref = p_ref.eval(pp);
    auto output_ref  = outputs_ref[0];
    lens    = output_ref.get_shape().lengths();
    elem_num = std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<std::size_t>());
    float* output_ref_data_ptr = reinterpret_cast<float*>(output_ref.data());
    std::vector<float> output_ref_vec(output_ref_data_ptr, output_ref_data_ptr + elem_num);
    // compare the results;
    CHECK(output_vec == output_ref_vec);
}

TEST_CASE(add_op)
{
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    auto y             = m.add_parameter("y", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    auto add_op        = migraphx::operation("add");
    auto r = m.add_instruction(add_op, {x, y});
    m.add_return({r});
    run_test_on_gpu(p);
}

TEST_CASE(if_then_else_op)
{
    migraphx::program p;
    auto mm = p.get_main_module();
    migraphx::shape cond_s{migraphx_shape_bool_type};
    auto cond = mm.add_parameter("cond", cond_s);

    auto then_mod = p.create_module("If_0_if");
    auto z        = then_mod.add_parameter("x", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    then_mod.add_return({z});

    auto else_mod = p.create_module("If_0_else");
    z             = else_mod.add_parameter("y", migraphx::shape(migraphx_shape_float_type, {3, 3}));
    else_mod.add_return({z});

    auto ret          = mm.add_instruction(migraphx::operation("if"), {cond}, {then_mod, else_mod});
    auto get_tuple_op = migraphx::operation("get_tuple_elem", "{index: 0}");
    mm.add_instruction(get_tuple_op, {ret});
    run_test_on_gpu(p);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
