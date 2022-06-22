#include "migraphx/instruction.hpp"
#include <iostream>
#include <vector>
#include <migraphx/gpu/target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

TEST_CASE(host_same_buffer_copy)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4, 2}};
    auto a           = mm->add_parameter("a", ss);
    auto b           = mm->add_parameter("b", ss);
    auto aa          = mm->add_instruction(migraphx::make_op("add"), a, a);
    auto gpu_out     = mm->add_instruction(migraphx::make_op("hip::copy_from_gpu"), aa);
    auto stream_sync = mm->add_instruction(migraphx::make_op("hip::sync_stream"), gpu_out);
    auto pass        = mm->add_instruction(unary_pass_op{}, stream_sync);
    auto alloc       = mm->add_instruction(
        migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ss)}, {"tag", ""}}));
    auto gpu_in = mm->add_instruction(migraphx::make_op("hip::copy_to_gpu"), pass, alloc);
    auto aab    = mm->add_instruction(migraphx::make_op("add"), gpu_in, b);
    mm->add_return({aab});
    migraphx::parameter_map pp;
    std::vector<float> a_vec(ss.elements(), -1);
    std::vector<float> b_vec(ss.elements(), 2);
    std::vector<float> c_vec(ss.elements(), 0);
    pp["a"] = migraphx::argument(ss, a_vec.data());
    pp["b"] = migraphx::argument(ss, b_vec.data());
    std::vector<float> gpu_result;
    migraphx::target gpu_t = migraphx::gpu::target{};
    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(gpu_t, options);
    auto result = p.eval(pp).back();
    std::vector<float> results_vector(ss.elements(), -1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(c_vec, results_vector));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
