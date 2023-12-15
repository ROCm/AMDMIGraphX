
#include <onnx_test.hpp>
#include <random>


TEST_CASE(randomuniformlike_test)
{
    float high = 10.0;
    float low  = 1.0;
    float seed = 0.0;
    std::vector<int> shape_attr{2, 3, 4};

    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::half_type, shape_attr};
    std::vector<double> rand_vals(s.elements());
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> d(low, high);
    std::generate(rand_vals.begin(), rand_vals.end(), [&]() { return d(gen); });

    mm->add_parameter("input", s);
    mm->add_literal(migraphx::literal{s, rand_vals});

    auto prog = optimize_onnx("randomuniformlike_test.onnx");

    EXPECT(p == prog);
}


