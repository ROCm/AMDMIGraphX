
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(mvn_default_axes_fp16_test)
{
    using migraphx::half;
    auto result =
        mvn_test<half>({2, 2, 2, 2}, migraphx::parse_onnx("mvn_default_axes_fp16_test.onnx"));
    std::vector<half> gold{half{-1.324},
                           half{-1.084},
                           half{-0.843},
                           half{-0.602},
                           half{-1.324},
                           half{-1.084},
                           half{-0.843},
                           half{-0.602},
                           half{0.602},
                           half{0.843},
                           half{1.084},
                           half{1.324},
                           half{0.602},
                           half{0.843},
                           half{1.084},
                           half{1.324}};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}
