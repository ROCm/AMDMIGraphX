
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(if_literal_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_literal_test.onnx");
        p.compile(migraphx::make_target("ref"));
        migraphx::shape s_data{migraphx::shape::bool_type};
        std::vector<char> data = {static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["cond"] = migraphx::argument(s_data, data.data());

        auto result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

        return result_vector;
    };

    // then branch
    {
        auto result_vector      = run_prog(true);
        std::vector<float> gold = {1, 2, 3, 4, 5};
        EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {5, 4, 3, 2, 1};
        EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    }
}


