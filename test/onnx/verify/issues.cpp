#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>
#include <migraphx/stringutils.hpp>

migraphx::shape make_shape(std::vector<size_t> lens, std::vector<size_t> strides = {})
{
    if(strides.empty())
        return migraphx::shape{migraphx::shape::float_type, lens};

    return migraphx::shape{migraphx::shape::float_type, lens, strides};
}

TEST_CASE(transpose_dot_issue)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x_sh = make_shape({2, 2, 1});
    auto x1   = mm->add_parameter("x1", x_sh);
    auto x2   = mm->add_parameter("x2", x_sh);
    x2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), x2);
    mm->add_instruction(migraphx::make_op("dot"), x1, x2);

    migraphx::compile_options opts;
    opts.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), opts);
    std::cout << p << std::endl;

    std::vector<float> x1_data(x_sh.elements());
    std::iota(x1_data.begin(), x1_data.end(), 0);
    std::vector<float> x2_data(x_sh.elements());
    std::iota(x2_data.begin(), x2_data.end(), x_sh.elements());
    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_sh, x1_data.data()};
    pm["x2"] = migraphx::argument{x_sh, x2_data.data()};

    auto res = p.eval(pm).back();
    std::vector<float> result_vector;
    res.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 4, 5, 12, 14, 18, 21};
    EXPECT(res.get_shape() == make_shape({2, 2, 2}));
    EXPECT(result_vector == gold);
}

TEST_CASE(broadcast_dot_issue)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x1_sh = make_shape({2, 2, 2});
    auto x2_sh = make_shape({2, 1, 2});
    auto x1    = mm->add_parameter("x1", x1_sh);
    auto x2    = mm->add_parameter("x2", x2_sh);
    x2 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x1_sh.lens()}}), x2);
    mm->add_instruction(migraphx::make_op("dot"), x1, x2);

    migraphx::compile_options opts;
    opts.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), opts);
    std::cout << p << std::endl;

    std::vector<float> x1_data(x1_sh.elements());
    std::iota(x1_data.begin(), x1_data.end(), 0);
    std::vector<float> x2_data(x2_sh.elements());
    std::iota(x2_data.begin(), x2_data.end(), x1_sh.elements());
    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_sh, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_sh, x2_data.data()};

    auto res = p.eval(pm).back();
    std::vector<float> result_vector;
    res.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{8, 9, 40, 45, 90, 99, 130, 143};
    EXPECT(res.get_shape() == make_shape({2, 2, 2}));
    EXPECT(result_vector == gold);
}
