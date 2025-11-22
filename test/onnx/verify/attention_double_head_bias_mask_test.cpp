/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/bf16.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(attention_double_head_bias_mask_batch1_passthrough_mask_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.836348f,
                                     0.3581245f,
                                     0.4475229f,
                                     0.5970712f,
                                     0.598615f,
                                     0.2951546f,
                                     0.2121896f,
                                     0.09142321};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.7183016f, 0.577417f,  0.3676456f, 0.878583f,  0.8927927f, 0.3989088f, 0.2036403f,
        0.4419216f, 0.508894f,  0.2171135f, 0.3508348f, 0.3410549f, 0.8376661f, 0.896282f,
        0.750656f,  0.9309627f, 0.1768383f, 0.831951f,  0.7498408f, 0.9859409f, 0.669683f,
        0.387659f,  0.397969f,  0.82948f,   0.776944f,  0.2565615f, 0.5149419f, 0.3952615f,
        0.467096f,  0.69897f,   0.764025f,  0.5253237f, 0.9064581f, 0.588643f,  0.1440694f,
        0.2109225f, 0.645562f,  0.7252325f, 0.972994f,  0.463554f,  0.9931276f, 0.6490803f,
        0.4276592f, 0.22789f,   0.7321448f, 0.89267f,   0.3418082f, 0.36943f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data(12, 0.0f);

    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {1, 1};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {
        1.380390f, 1.002805f, 0.6146411f, 0.791859f, 1.28671f, 0.919257f, 0.5846517f, 0.756680f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_double_head_bias_mask_batch1_last_mask_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {
        0.2365631f, 0.2897484f, 0.461962f, 0.547217f, 0.0343506f, 0.217421f, 0.2617837f, 0.642011f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.740227f,  0.4794638f, 0.7512885f, 0.850956f,  0.1834944f, 0.0166223f, 0.506032f,
        0.307133f,  0.809147f,  0.4110304f, 0.0074044f, 0.4610349f, 0.4528455f, 0.7661606f,
        0.435064f,  0.88811f,   0.424238f,  0.2526340f, 0.3349141f, 0.896244f,  0.2684531f,
        0.4195141f, 0.234627f,  0.3939499f, 0.5542862f, 0.668919f,  0.6118851f, 0.4075400f,
        0.2731890f, 0.632885f,  0.3904807f, 0.5439004f, 0.118005f,  0.625134f,  0.4839466f,
        0.3752088f, 0.5442267f, 0.812204f,  0.687263f,  0.1925360f, 0.822273f,  0.8223247f,
        0.9237169f, 0.961031f,  0.1192676f, 0.88399f,   0.7545891f, 0.42134193};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.561730f,
                                    0.3377643f,
                                    0.2519498f,
                                    0.2072306f,
                                    0.5650865f,
                                    0.347968f,
                                    0.1985114f,
                                    0.6656775f,
                                    0.3533229f,
                                    0.5744964f,
                                    0.5860762f,
                                    0.76156753f};

    // 0 = mask,1  = pass through
    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {1, 0};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {
        0.7423008f, 1.565812f, 1.292300f, 1.388675f, 0.7423008f, 1.565812f, 1.292300f, 1.388675f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_double_head_bias_mask_batch1_first_mask_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.4530325f,
                                     0.0649147f,
                                     0.127795f,
                                     0.3183984f,
                                     0.589117f,
                                     0.5000232f,
                                     0.3351452f,
                                     0.15613051f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.778988f,  0.7876636f, 0.0258449f, 0.4360487f, 0.131283f,  0.859008f,  0.985208f,
        0.3964851f, 0.4431120f, 0.714212f,  0.7847860f, 0.5370770f, 0.952801f,  0.7038981f,
        0.593976f,  0.3858227f, 0.2331966f, 0.7834450f, 0.914282f,  0.923162f,  0.612781f,
        0.0974504f, 0.6551792f, 0.0908233f, 0.2610949f, 0.5107522f, 0.2366336f, 0.1124516f,
        0.0359591f, 0.7155804f, 0.643442f,  0.5108542f, 0.9312866f, 0.379560f,  0.244760f,
        0.86605f,   0.0146525f, 0.3647071f, 0.6877476f, 0.651158f,  0.3327819f, 0.4332133f,
        0.0284249f, 0.2950900f, 0.868766f,  0.167157f,  0.5624779f, 0.96631414f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.71639f,
                                    0.6984057f,
                                    0.641569f,
                                    0.5584603f,
                                    0.807863f,
                                    0.9522143f,
                                    0.964991f,
                                    0.0536422f,
                                    0.6487126f,
                                    0.924860f,
                                    0.73642f,
                                    0.85529f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {0, 1};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {
        1.6639f, 1.547649f, 1.696211f, 1.658239f, 1.6639f, 1.547649f, 1.696211f, 1.6582391f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_double_head_bias_mask_batch1_all_mask_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_batch1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {
        0.550855f, 0.9449020f, 0.4138677f, 0.609427f, 0.0278122f, 0.929195f, 0.72592f, 0.98518866f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.1919869f, 0.97087f,   0.0650908f, 0.4648978f, 0.784916f,  0.866638f,  0.9574522f,
        0.618995f,  0.333331f,  0.465269f,  0.945676f,  0.1423377f, 0.913690f,  0.472937f,
        0.739001f,  0.3276381f, 0.8793762f, 0.883888f,  0.7565744f, 0.607640f,  0.418141f,
        0.471184f,  0.4038756f, 0.893745f,  0.445011f,  0.929533f,  0.0949846f, 0.690924f,
        0.9738619f, 0.822521f,  0.4315533f, 0.884698f,  0.0233254f, 0.5688004f, 0.3722963f,
        0.3832031f, 0.506166f,  0.1660704f, 0.4618730f, 0.0794496f, 0.0844472f, 0.6387809f,
        0.7633692f, 0.7961660f, 0.786498f,  0.71053f,   0.344582f,  0.44591686f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.751496f,
                                    0.557292f,
                                    0.6720010f,
                                    0.1879267f,
                                    0.352546f,
                                    0.600021f,
                                    0.0552079f,
                                    0.5959239f,
                                    0.0404032f,
                                    0.1882552f,
                                    0.2718655f,
                                    0.84921235f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {1, 2}};
    std::vector<float> mask_data = {0, 0};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {
        1.166096f, 1.650388f, 1.406106f, 2.30548f, 1.165592f, 1.649586f, 1.406728f, 2.304997};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_double_head_bias_mask_batch2_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.82405f,
                                     0.4764252f,
                                     0.377304f,
                                     0.4109286f,
                                     0.2003798f,
                                     0.767443f,
                                     0.0175668f,
                                     0.7279092f,
                                     0.294912f,
                                     0.774890f,
                                     0.8893084f,
                                     0.71005f,
                                     0.2514481f,
                                     0.768277f,
                                     0.3042429f,
                                     0.47951823f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.1694838f, 0.9370f,    0.0493409f, 0.0579776f, 0.4779790f, 0.121594f,  0.4990379f,
        0.224993f,  0.7370581f, 0.2758335f, 0.6196964f, 0.2212499f, 0.2567361f, 0.9864421f,
        0.1344247f, 0.0343491f, 0.997883f,  0.0671764f, 0.745450f,  0.0054291f, 0.507925f,
        0.6584495f, 0.576835f,  0.5474540f, 0.3751728f, 0.820518f,  0.0764405f, 0.9172129f,
        0.3914677f, 0.1670402f, 0.651346f,  0.0744788f, 0.9613833f, 0.1876087f, 0.5287391f,
        0.1745295f, 0.4547757f, 0.005437f,  0.267799f,  0.709826f,  0.1078507f, 0.603933f,
        0.50390f,   0.2069911f, 0.4061629f, 0.0048359f, 0.3187260f, 0.13678697};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.78651f,
                                    0.809411f,
                                    0.982877f,
                                    0.3211297f,
                                    0.0238379f,
                                    0.640969f,
                                    0.9006f,
                                    0.1837699f,
                                    0.529567f,
                                    0.927664f,
                                    0.5510397f,
                                    0.83173823f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<float> mask_data = {0, 0, 0, 0};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {1.642277f,
                               1.518098f,
                               1.52985f,
                               1.397790f,
                               1.646151f,
                               1.518438f,
                               1.529783f,
                               1.397790f,
                               2.06642f,
                               1.649505f,
                               1.733022f,
                               1.522298f,
                               2.042337f,
                               1.645073f,
                               1.724378f,
                               1.5192288f};

    // Adjust tolerance on this since "all mask" case is not common but also likely due to use
    // adding -10000 to numbers and then doing the softmax , resulting in loss of precision
    // Other mask values seem to be valid from 1e-5 range
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold, 200));
}

TEST_CASE(attention_double_head_bias_mask_batch2_all_pass_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.2773895f,
                                     0.5486851f,
                                     0.451570f,
                                     0.99696f,
                                     0.406118f,
                                     0.4839315f,
                                     0.755350f,
                                     0.8436214f,
                                     0.6108430f,
                                     0.1231699f,
                                     0.46070f,
                                     0.7012386f,
                                     0.759620f,
                                     0.2762067f,
                                     0.7038193f,
                                     0.569161f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.3845310f, 0.9619347f, 0.4936712f, 0.6283104f, 0.841628f,  0.350351f,  0.4579307f,
        0.2874027f, 0.2029373f, 0.4477961f, 0.2471958f, 0.624741f,  0.3650294f, 0.4590854f,
        0.2959658f, 0.4803923f, 0.9737639f, 0.4497354f, 0.830746f,  0.3309565f, 0.4202715f,
        0.930299f,  0.998847f,  0.9334816f, 0.8441676f, 0.52829f,   0.4335649f, 0.2783789f,
        0.9052453f, 0.0463431f, 0.975376f,  0.6249313f, 0.8802423f, 0.699552f,  0.638825f,
        0.959790f,  0.8345969f, 0.1295447f, 0.616319f,  0.433175f,  0.0142569f, 0.4430008f,
        0.1237481f, 0.25647f,   0.369231f,  0.5137827f, 0.0363198f, 0.13236965f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.612893f,
                                    0.777793f,
                                    0.3126879f,
                                    0.4798205f,
                                    0.776248f,
                                    0.5968348f,
                                    0.459095f,
                                    0.1061234f,
                                    0.807113f,
                                    0.3901378f,
                                    0.5912092f,
                                    0.04816633f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<float> mask_data = {1, 1, 1, 1};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {1.986912f,
                               1.932524f,
                               1.627824f,
                               1.477383f,
                               1.988329f,
                               1.93341f,
                               1.628718f,
                               1.479056f,
                               1.823226f,
                               1.671619f,
                               1.403065f,
                               1.369300f,
                               1.828217f,
                               1.677599f,
                               1.408375f,
                               1.3763521f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_double_head_bias_mask_batch2_diag_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.872664f,
                                     0.5373890f,
                                     0.4955406f,
                                     0.3421586f,
                                     0.2306984f,
                                     0.826305f,
                                     0.4227518f,
                                     0.567517f,
                                     0.1019400f,
                                     0.8496197f,
                                     0.895266f,
                                     0.136645f,
                                     0.9690335f,
                                     0.5001573f,
                                     0.932205f,
                                     0.67461586f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.0345085f, 0.4419922f, 0.5818482f, 0.2626504f, 0.348432f,  0.0842294f, 0.182248f,
        0.968666f,  0.418183f,  0.0105558f, 0.6986581f, 0.1786073f, 0.1195501f, 0.4313392f,
        0.6098367f, 0.662050f,  0.469720f,  0.0641924f, 0.1400848f, 0.8684591f, 0.4107752f,
        0.2899719f, 0.7876609f, 0.634067f,  0.367954f,  0.1173279f, 0.858620f,  0.663137f,
        0.2342181f, 0.1797239f, 0.918695f,  0.4486411f, 0.0721015f, 0.731420f,  0.769667f,
        0.617797f,  0.2715653f, 0.6987223f, 0.0972901f, 0.2079825f, 0.0658583f, 0.330616f,
        0.3264711f, 0.4618025f, 0.68533f,   0.0782526f, 0.0477516f, 0.08156187f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.9885801f,
                                    0.2215420f,
                                    0.4647936f,
                                    0.629216f,
                                    0.1154782f,
                                    0.5222471f,
                                    0.734802f,
                                    0.2114844f,
                                    0.333944f,
                                    0.2471313f,
                                    0.7204917f,
                                    0.433187f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<float> mask_data = {1, 0, 0, 1};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {1.189846f,
                               0.801394f,
                               2.151206f,
                               1.26384f,
                               1.189846f,
                               0.801394f,
                               2.151206f,
                               1.26384f,
                               1.474180f,
                               1.137016f,
                               2.541171f,
                               1.554334f,
                               1.474180f,
                               1.137016f,
                               2.541171f,
                               1.5543344f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(attention_double_head_bias_mask_batch2_off_diag_test)
{
    auto p = optimize_onnx("attention_double_head_bias_mask_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 2, 4}};
    // Taken from attention_op_test.cc from Onnxruntime repo
    std::vector<float> input_data = {0.2540857f,
                                     0.0256114f,
                                     0.6211387f,
                                     0.4503982f,
                                     0.4017042f,
                                     0.555549f,
                                     0.896553f,
                                     0.4662611f,
                                     0.179541f,
                                     0.0961751f,
                                     0.715316f,
                                     0.2744848f,
                                     0.1165528f,
                                     0.790455f,
                                     0.5210529f,
                                     0.30495727f};

    migraphx::shape weight_shape{migraphx::shape::float_type, {4, 12}};
    std::vector<float> weight_data = {
        0.3110327f, 0.854108f,  0.816163f,  0.960607f,  0.004508f,  0.4068033f, 0.9540065f,
        0.975203f,  0.9766478f, 0.9929510f, 0.9142452f, 0.4771942f, 0.1567466f, 0.2185167f,
        0.0457491f, 0.773614f,  0.6616419f, 0.595781f,  0.3763947f, 0.3090001f, 0.1318927f,
        0.373421f,  0.3436092f, 0.362673f,  0.8083966f, 0.5987018f, 0.0699282f, 0.8792643f,
        0.910243f,  0.0060881f, 0.4817900f, 0.5833030f, 0.5045705f, 0.2815545f, 0.904394f,
        0.9449780f, 0.4684918f, 0.1758799f, 0.0536868f, 0.6681251f, 0.5750992f, 0.747650f,
        0.809778f,  0.899023f,  0.371502f,  0.1342116f, 0.4682391f, 0.886880f};

    migraphx::shape bias_shape{migraphx::shape::float_type, {12}};
    std::vector<float> bias_data = {0.0209900f,
                                    0.8722936f,
                                    0.2948170f,
                                    0.689064f,
                                    0.1642694f,
                                    0.758328f,
                                    0.606924f,
                                    0.099455f,
                                    0.4234262f,
                                    0.2398736f,
                                    0.692330f,
                                    0.950954f};

    migraphx::shape mask_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<float> mask_data = {0, 1, 1, 0};

    migraphx::literal input{input_shape, input_data};
    migraphx::literal weights{weight_shape, weight_data};
    migraphx::literal bias{bias_shape, bias_data};
    migraphx::literal mask{mask_shape, mask_data};

    pp["input"]      = input.get_argument();
    pp["weights"]    = weights.get_argument();
    pp["bias"]       = bias.get_argument();
    pp["mask_index"] = mask.get_argument();

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold data from AttentionNoMaskIndex from attention_op_test.cc from Onnxruntime
    std::vector<float> gold = {1.514614f,
                               1.161206f,
                               2.279638f,
                               2.6048f,
                               1.514614f,
                               1.161206f,
                               2.279638f,
                               2.6048f,
                               1.074359f,
                               0.692303f,
                               1.66497f,
                               1.9909048f,
                               1.074359f,
                               0.692303f,
                               1.66497f,
                               1.9909048f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
