
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>


TEST_CASE(mvn_default_axes_test)
{
    auto result = mvn_test({2, 2, 2, 2}, migraphx::parse_onnx("mvn_default_axes_test.onnx"));
    std::vector<float> gold{-1.32424438,
                            -1.08347268,
                            -0.84270097,
                            -0.60192927,
                            -1.32424438,
                            -1.08347268,
                            -0.84270097,
                            -0.60192927,
                            0.60192927,
                            0.84270097,
                            1.08347268,
                            1.32424438,
                            0.60192927,
                            0.84270097,
                            1.08347268,
                            1.32424438};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}


