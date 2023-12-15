
#include <onnx_test.hpp>
#include <random>

TEST_CASE(randomnormal_test)
{
    float mean  = 10.0;
    float scale = 1.5;
    float seed  = 0.0;
    std::vector<int> shape_attr{2, 3, 4};

    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::double_type, shape_attr};
    std::vector<double> rand_vals(s.elements());
    std::mt19937 gen(seed);
    std::normal_distribution<> d(mean, scale);
    std::generate(rand_vals.begin(), rand_vals.end(), [&]() { return d(gen); });

    mm->add_literal(migraphx::literal{s, rand_vals});

    auto prog = optimize_onnx("randomnormal_test.onnx");

    EXPECT(p == prog);
}
