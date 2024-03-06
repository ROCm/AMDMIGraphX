/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(einsum_permute_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_permute_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.06727745, 0.21160052, 0.1340474, 0.74153227, 0.40337096, 0.81284493};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.06727745, 0.74153227, 0.21160052, 0.40337096, 0.1340474, 0.81284493};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_summation_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_summation_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.79413969, 0.45169144, 0.06846618, 0.67973967, 0.83375529, 0.44838823};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {3.2761804984270566};
    EXPECT(result_vector == gold);
}

TEST_CASE(einsum_column_sum_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_column_sum_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.22235926, 0.83263138, 0.04747776, 0.96030827, 0.18947713, 0.48815767};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.18266753, 1.0221085, 0.53563543};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_row_sum_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_row_sum_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.17123185, 0.59008514, 0.37948294, 0.73022965, 0.22919172, 0.27532941};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.14079993, 1.23475077};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_vector_multiplication_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_vector_multiplication_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.4834133, 0.14106742, 0.50055824, 0.91764271, 0.95528452, 0.98199955};

    migraphx::shape v_shape{migraphx::shape::float_type, {1, 3}};
    std::vector<float> v_data = {0.73961958, 0.53071864, 0.34152803};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};
    pm["v"] = migraphx::argument{v_shape, v_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.60336371, 1.52107419};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_matrix_multiplication_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_matrix_multiplication_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.45176257, 0.84846429, 0.4374105, 0.25132236, 0.70519571, 0.4902031};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_shape, x_data.data()};
    pm["x2"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.11530901, 0.92629139, 0.92629139, 0.80076299};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_vector_dot_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_vector_dot_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {3}};
    std::vector<float> x_data = {0.45263196, 0.90876706, 0.9584567};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_shape, x_data.data()};
    pm["x2"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.94937252};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_dot_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_dot_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.50001808, 0.12468059, 0.85439214, 0.00773521, 0.84764693, 0.87185525};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_shape, x_data.data()};
    pm["x2"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {2.47424599};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_hadamard_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_hadamard_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.86162928, 0.76609605, 0.03362172, 0.21778614, 0.27204858, 0.83778314};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_shape, x_data.data()};
    pm["x2"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.74240502, 0.58690315, 0.00113042, 0.0474308, 0.07401043, 0.70188058};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_vector_outer_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_vector_outer_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {3}};
    std::vector<float> x1_data = {0.35935151, 0.51298139, 0.46076789};

    migraphx::shape x2_shape{migraphx::shape::float_type, {5}};
    std::vector<float> x2_data = {0.82417482, 0.17984153, 0.17680769, 0.55499376, 0.74447638};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.29616847,
                               0.06462632,
                               0.06353611,
                               0.19943785,
                               0.26752871,
                               0.42278634,
                               0.09225536,
                               0.09069905,
                               0.28470147,
                               0.38190252,
                               0.37975329,
                               0.0828652,
                               0.08146731,
                               0.2557233,
                               0.34303081};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_outer_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_outer_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x1_data = {
        0.25870501, 0.06755926, 0.18247427, 0.19436556, 0.61580192, 0.20010939};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 5}};
    std::vector<float> x2_data = {0.30771264,
                                  0.86270274,
                                  0.55251869,
                                  0.35880608,
                                  0.3234085,
                                  0.24642323,
                                  0.82411907,
                                  0.33488431,
                                  0.69288027,
                                  0.21717812};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.0796068,  0.22318552, 0.14293935, 0.09282493, 0.0836674,  0.06375092, 0.21320373,
        0.08663625, 0.17925159, 0.05618507, 0.02078884, 0.05828356, 0.03732775, 0.02424067,
        0.02184924, 0.01664817, 0.05567687, 0.02262453, 0.04681048, 0.01467239, 0.05614964,
        0.15742105, 0.10082044, 0.06547288, 0.05901373, 0.0449659,  0.15038052, 0.06110777,
        0.12643282, 0.03962942, 0.05980874, 0.1676797,  0.1073906,  0.06973954, 0.06285947,
        0.04789619, 0.16018036, 0.06508997, 0.13467206, 0.04221195, 0.18949004, 0.53125401,
        0.34024207, 0.22095347, 0.19915557, 0.1517479,  0.50749411, 0.2062224,  0.426677,
        0.1337387,  0.06157619, 0.17263492, 0.11056418, 0.07180047, 0.06471708, 0.0493116,
        0.16491396, 0.06701349, 0.13865185, 0.04345938};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_batch_matrix_multiplication_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_batch_matrix_multiplication_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {3, 2, 5}};
    std::vector<float> x1_data = {0.99236023, 0.6848901,  0.37916487, 0.35448254, 0.06103943,
                                  0.88991707, 0.20816843, 0.12124124, 0.90632983, 0.88490338,
                                  0.93530363, 0.41393917, 0.95269137, 0.95556378, 0.63113954,
                                  0.87936215, 0.66831395, 0.38079353, 0.74128241, 0.05493966,
                                  0.12545692, 0.77418839, 0.17562823, 0.5558762,  0.95698858,
                                  0.49207445, 0.81934147, 0.50168285, 0.13782384, 0.71351839};

    migraphx::shape x2_shape{migraphx::shape::float_type, {3, 5, 3}};
    std::vector<float> x2_data = {
        0.72870257, 0.44635711, 0.05938103, 0.7031737,  0.52116502, 0.01719079, 0.99837568,
        0.29989025, 0.63673246, 0.39255282, 0.39796917, 0.03082538, 0.20994321, 0.11431396,
        0.06561894, 0.99749458, 0.45970296, 0.76957234, 0.98073012, 0.63154904, 0.22862209,
        0.71098086, 0.68895963, 0.92763041, 0.61730666, 0.54453456, 0.99719059, 0.05984043,
        0.64232788, 0.9754334,  0.39450223, 0.1005812,  0.11753032, 0.59885466, 0.75932222,
        0.45269589, 0.26201765, 0.39022748, 0.96507247, 0.55260731, 0.42233854, 0.50671452,
        0.60313192, 0.32628192, 0.40066181};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.73524908,
                               1.06164644,
                               0.32706016,
                               1.45746952,
                               1.00391812,
                               0.21962538,
                               2.64391179,
                               2.27348666,
                               3.26667873,
                               2.26421769,
                               1.52761296,
                               1.97554961,
                               1.44350867,
                               1.21602803,
                               1.19981019,
                               1.32274886,
                               1.15842452,
                               1.2686234};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_tensor_contraction_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_tensor_contraction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 3, 5, 7}};
    std::vector<float> x1_data = {0.95685496, 0.40756636, 0.65360334, 0.96968506,
                                  0.50135366, 0.50255377, 0.54263245, 0.40919774,
                                  0.0512559,  0.18771721, 0.79265052, 0.76059609,
                                  0.31619353, 0.62297555, 0.70398181, 0.82378161,
                                  0.50388425, 0.56257752, 0.29233331, 0.98995162,
                                  0.38240504, 0.29803141, 0.23344604, 0.78356941,
                                  0.67958479, 0.10005701, 0.15588056, 0.29163352,
                                  0.90480928, 0.35649064, 0.77419322, 0.56301202,
                                  0.133201,   0.33165803, 0.37175546, 0.63959881,
                                  0.6058814,  0.43169871, 0.65272681, 0.17943427,
                                  0.30863453, 0.39029972, 0.66189176, 0.2311467,
                                  0.77007359, 0.33601537, 0.28087721, 0.65732174,
                                  0.67537887, 0.65066593, 0.89716601, 0.92921684,
                                  0.69368177, 0.86772161, 0.82583412, 0.32274594,
                                  0.0739795,  0.7573278,  0.82209441, 0.44979001,
                                  0.52619926, 0.68870551, 0.8586619,  0.32302478,
                                  0.30437449, 0.22181276, 0.41919667, 0.16351355,
                                  0.10825966, 0.20406509, 0.32577585, 0.89748513,
                                  0.78650319, 0.55487763, 0.74600253, 0.68125503,
                                  0.59796741, 0.75181214, 0.27655496, 0.87750203,
                                  0.50401991, 0.30561784, 0.82724439, 0.04727558,
                                  0.9224091,  0.24823561, 0.05547919, 0.93431458,
                                  0.51550858, 0.64800403, 0.95942825, 0.04009098,
                                  0.55616792, 0.71433063, 0.0753035,  0.0479713,
                                  0.19538077, 0.29627466, 0.47649694, 0.49999562,
                                  0.05246693, 0.29663604, 0.29992186, 0.62328915,
                                  0.00265317, 0.50642525, 0.73613139, 0.5998967,
                                  0.37132279, 0.02788106, 0.99984085, 0.87220473,
                                  0.08963238, 0.20698509, 0.17961793, 0.32962012,
                                  0.8046416,  0.96530006, 0.27079326, 0.07223538,
                                  0.72336279, 0.54842596, 0.38904735, 0.21660217,
                                  0.05165004, 0.60308648, 0.98992912, 0.01950237,
                                  0.19094762, 0.2928557,  0.18129261, 0.23948649,
                                  0.65970424, 0.0217831,  0.89637346, 0.25872699,
                                  0.98701943, 0.43783966, 0.65803132, 0.06773888,
                                  0.11277457, 0.68990466, 0.80914248, 0.66815968,
                                  0.10671669, 0.15578704, 0.78813393, 0.71601124,
                                  0.41304412, 0.93551562, 0.28607031, 0.16353775,
                                  0.54597636, 0.10405413, 0.05332971, 0.8301183,
                                  0.0991274,  0.1152268,  0.86477572, 0.20824363,
                                  0.77115011, 0.62202978, 0.87562719, 0.17638816,
                                  0.00798768, 0.46176706, 0.33432177, 0.93926911,
                                  0.60557399, 0.38483151, 0.23797486, 0.83815198,
                                  0.27293845, 0.62067518, 0.56702013, 0.80762545,
                                  0.47669687, 0.13692723, 0.40838777, 0.3148337,
                                  0.55255245, 0.24319153, 0.39330312, 0.22781179,
                                  0.101221,   0.80367016, 0.08707603, 0.90069816,
                                  0.28595044, 0.57599756, 0.71276499, 0.04032091,
                                  0.50101916, 0.94582167, 0.2091183,  0.17698968,
                                  0.72687874, 0.08878026, 0.16422912, 0.34543801,
                                  0.28480515, 0.8740834,  0.18413319, 0.60564407,
                                  0.94070861, 0.21143538, 0.2715485,  0.76848231,
                                  0.0064918,  0.36614132

    };

    migraphx::shape x2_shape{migraphx::shape::float_type, {1, 3, 3, 7, 5}};
    std::vector<float> x2_data = {
        0.31719105, 0.44506343, 0.59957066, 0.00373946, 0.06497482, 0.30887562, 0.04364479,
        0.09203816, 0.0778086,  0.58357676, 0.49651904, 0.10000999, 0.16565024, 0.46539611,
        0.82516851, 0.64563229, 0.26637135, 0.2141455,  0.69189904, 0.75060041, 0.75433425,
        0.69215069, 0.18186255, 0.89800939, 0.93269204, 0.63033347, 0.9423835,  0.90530682,
        0.07135205, 0.57649693, 0.44479805, 0.94513207, 0.89856664, 0.79120729, 0.63383186,
        0.97271015, 0.69211656, 0.91893391, 0.07601606, 0.90099522, 0.31441974, 0.70932527,
        0.68997715, 0.33528514, 0.24921017, 0.09703337, 0.54714714, 0.98431729, 0.27753988,
        0.78936545, 0.51031898, 0.30604168, 0.53546681, 0.95644451, 0.79345859, 0.3444766,
        0.19356174, 0.41127976, 0.15782141, 0.65660564, 0.76540504, 0.21572256, 0.29864542,
        0.01153175, 0.06708682, 0.82473386, 0.45034386, 0.96212735, 0.5969872,  0.35962495,
        0.60466663, 0.52630816, 0.73655946, 0.11649375, 0.32456538, 0.64199728, 0.08340919,
        0.2237889,  0.09521117, 0.91767416, 0.22842615, 0.46863323, 0.00293057, 0.13495504,
        0.68305119, 0.80013148, 0.24702202, 0.83619373, 0.94419611, 0.25176846, 0.74292949,
        0.68404465, 0.23097011, 0.09664962, 0.44346347, 0.31467353, 0.37099949, 0.54412241,
        0.76552126, 0.1443158,  0.03555697, 0.43584746, 0.10575715, 0.1046359,  0.43291613,
        0.03007743, 0.55544576, 0.80022343, 0.42529416, 0.47484557, 0.84443037, 0.99362024,
        0.78040286, 0.16341681, 0.98059931, 0.64114384, 0.27438947, 0.51972672, 0.24844974,
        0.11630196, 0.86696682, 0.62380654, 0.23221499, 0.93125653, 0.53386878, 0.14323035,
        0.46524576, 0.24347234, 0.43592108, 0.68938894, 0.83452471, 0.67473429, 0.11704585,
        0.01223517, 0.61133307, 0.19640497, 0.94062148, 0.09548036, 0.27914148, 0.28533241,
        0.32062872, 0.27619432, 0.18284111, 0.73646915, 0.07043039, 0.10841211, 0.25284529,
        0.73262578, 0.63395762, 0.75505585, 0.66397536, 0.60934204, 0.17561379, 0.44185177,
        0.90064761, 0.87593443, 0.04697443, 0.90844936, 0.4878133,  0.17061924, 0.37868238,
        0.03991319, 0.99918374, 0.05644218, 0.11533688, 0.36478255, 0.74207249, 0.02537966,
        0.73720329, 0.41510019, 0.87408442, 0.0902388,  0.77849296, 0.22027469, 0.66811554,
        0.535826,   0.40478544, 0.47295354, 0.53722756, 0.81697433, 0.17400588, 0.52628511,
        0.57033592, 0.74645826, 0.58147372, 0.25898702, 0.03268815, 0.37127404, 0.04316943,
        0.86187713, 0.33330374, 0.58282901, 0.32484663, 0.8295674,  0.34023535, 0.48430125,
        0.5626468,  0.48469659, 0.16184832, 0.71399316, 0.5417521,  0.11897383, 0.84953376,
        0.98761605, 0.58273874, 0.89537346, 0.83282794, 0.78849938, 0.42528756, 0.08624209,
        0.7689597,  0.92518944, 0.25278458, 0.0732656,  0.0057378,  0.74097687, 0.13263284,
        0.73757523, 0.01510422, 0.8650508,  0.21755823, 0.38417346, 0.77236815, 0.80464568,
        0.23389132, 0.24982259, 0.3034747,  0.99357576, 0.69974824, 0.62271656, 0.43386392,
        0.3517672,  0.01739671, 0.54493487, 0.07725586, 0.75756086, 0.86409372, 0.50906544,
        0.87797418, 0.41355064, 0.11812738, 0.9809903,  0.67759122, 0.44601677, 0.53664097,
        0.75512155, 0.27589464, 0.12141359, 0.74533628, 0.95179317, 0.31788316, 0.41200016,
        0.81161753, 0.84035926, 0.42866542, 0.97692811, 0.14777789, 0.54256825, 0.03691842,
        0.71298109, 0.27676914, 0.31342084, 0.09905633, 0.01056144, 0.28488026, 0.39330704,
        0.07871612, 0.61847332, 0.48494692, 0.14455078, 0.53627478, 0.78087393, 0.24899241,
        0.78534409, 0.29844719, 0.33439453, 0.62448919, 0.21187341, 0.21381023, 0.25570138,
        0.67919933, 0.73611559, 0.45109776, 0.25360901, 0.17702297, 0.41635495, 0.80213947,
        0.01236559, 0.0112422,  0.03389217, 0.87942468, 0.25273501, 0.511234,   0.82734509,
        0.58747506, 0.31687443, 0.89906645, 0.96090575, 0.04004779, 0.02298561, 0.10433042,
        0.7104134,  0.79670464, 0.9930637,  0.5446879,  0.06004139, 0.41158374, 0.17676018,
        0.10056314, 0.01345726, 0.82521847, 0.76125409, 0.17694037, 0.05363529, 0.32265118};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        4.37385737, 3.07363193, 3.61847664, 4.34283839, 4.26894546, 4.00093768, 4.51345157,
        3.28585485, 4.98955956, 3.40062413, 4.32430907, 3.58727315, 4.01024983, 4.04214073,
        3.71183284, 4.04117845, 3.9304425,  4.05446572, 3.19462145, 3.75153593, 3.63370359,
        3.6737565,  2.89999382, 3.0174889,  4.35349886, 3.165444,   3.10185148, 3.86251195,
        3.3873455,  4.06622752, 2.90101219, 3.93475191, 3.00084537, 3.36253104, 3.50215565,
        3.2272778,  3.63297086, 4.11360191, 2.55025226, 2.89909597, 2.8134455,  2.91506006,
        3.7938589,  3.12994095, 3.93469812, 5.19912284, 4.38534872, 3.50334177, 4.71274384,
        3.59957887, 4.82387001, 2.82827241, 5.04315375, 3.42817516, 3.97827684, 4.0792739,
        3.73622444, 4.59885202, 4.20690004, 3.39733812, 3.56861724, 4.18875149, 3.80445766,
        4.34760619, 2.83154296, 3.39897749, 4.91619741, 4.55085299, 4.02356989, 4.83137925,
        3.49172193, 5.09758452, 3.46814603, 4.8534725,  3.58561246, 4.17459184, 4.57103074,
        4.31924652, 3.86027525, 4.33725934, 3.88334716, 3.51074837, 4.2163728,  3.76365513,
        3.13004972, 2.27159717, 2.35669807, 3.25755431, 2.85534261, 2.56412151, 3.19951963,
        2.50814311, 3.53231318, 2.20002443, 3.12059903, 2.63204045, 2.90076584, 3.36582992,
        3.06683373, 2.76686275, 2.77506122, 2.09060484, 2.37978869, 2.59300135, 2.73194814,
        4.12941618, 3.09876995, 3.26773346, 4.15566501, 3.49722972, 3.46654242, 4.2842499,
        3.77358659, 4.61660476, 3.14276911, 3.88478492, 3.36244681, 3.70141846, 3.77154536,
        3.59743975, 4.07663608, 3.81503321, 3.53650377, 3.19912915, 3.41346893, 3.6696098,
        3.22521498, 2.26604057, 2.16539957, 4.2136737,  2.91410526, 3.02978768, 3.33819415,
        2.9409972,  3.83464087, 2.65153712, 3.32360785, 2.24438948, 3.95703137, 3.35290512,
        3.41760415, 2.86825506, 3.08274974, 2.72484017, 2.65706605, 3.36092398, 2.83630318,
        2.89697041, 2.50152336, 2.73918816, 4.5120665,  3.40255688, 2.21408714, 2.82712268,
        3.04826657, 3.41090928, 2.96534728, 3.52745057, 2.24957446, 3.84521048, 3.08574989,
        3.28188229, 2.31822221, 3.76298328, 2.57778028, 3.19081461, 3.07155158, 2.73609241,
        4.19950589, 3.6560231,  3.78387066, 4.79181063, 3.83391543, 3.55914169, 4.5795992,
        3.80991087, 5.12966262, 3.81299104, 4.21955081, 3.59584019, 4.29810986, 3.70353926,
        3.70364291, 4.26908068, 3.98312417, 3.12472346, 3.16217195, 3.4642648,  3.22122407,
        2.62355294, 1.82932863, 1.87920164, 2.36533037, 2.06395846, 2.33422825, 2.78131656,
        1.83772458, 2.43196754, 2.45650722, 2.37074638, 1.36516771, 2.47311739, 1.85973378,
        2.28547527, 2.22058881, 2.42265217, 1.82521576, 1.42674238, 2.63853633, 2.09125692,
        3.43987729, 2.19115419, 2.93461373, 3.85600443, 3.76977612, 3.15357479, 3.3520207,
        2.6665599,  4.023041,   2.68187355, 3.41405847, 2.72865504, 3.23944437, 3.64514952,
        3.347772,   3.08780622, 3.59354671, 3.2772289,  2.50492638, 2.77853552, 3.07724088,
        3.03408917, 2.45574117, 2.5493586,  3.48528482, 2.74493899, 2.611099,   3.26765525,
        2.93502233, 3.93585413, 2.32960219, 3.09824088, 3.03519943, 3.21090064, 3.3114777,
        2.58394431, 2.2187237,  3.00954904, 2.23092399, 2.83426168, 2.27217761, 2.5014613,
        3.19291058, 2.17091072, 3.02885277, 4.41008881, 4.12811972, 3.61970552, 3.53615268,
        2.78509447, 4.861919,   2.54172549, 4.17995171, 2.56407684, 4.31953876, 3.98183007,
        4.18525975, 3.4355,     3.32306034, 2.80758129, 3.17616352, 3.6386068,  3.45497304,
        3.46339678, 2.31062665, 2.98872364, 4.14619218, 3.33730406, 2.814647,   4.28392461,
        2.85391039, 3.99487077, 3.22812695, 4.24891978, 2.57924025, 3.05409494, 3.2767709,
        3.64664984, 3.49454643, 3.69300505, 2.42169066, 2.93327166, 3.5987843,  2.52333694};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_diagonal_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_diagonal_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> x_data = {0.47776573,
                                 0.63448645,
                                 0.89651875,
                                 0.23679368,
                                 0.99918665,
                                 0.27613904,
                                 0.57251725,
                                 0.30676534,
                                 0.01097199};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.47776573, 0.99918665, 0.01097199};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_trace_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_trace_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> x_data = {0.90812557,
                                 0.40719192,
                                 0.71678312,
                                 0.78176503,
                                 0.57731702,
                                 0.23585615,
                                 0.06292936,
                                 0.46016886,
                                 0.37753559};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.86297818};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_2d_3d_multiplication_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_2d_3d_multiplication_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> x1_data = {0.77117604,
                                  0.10042859,
                                  0.68555583,
                                  0.93192629,
                                  0.39255794,
                                  0.99285767,
                                  0.88129697,
                                  0.56599014,
                                  0.03828527};

    migraphx::shape x2_shape{migraphx::shape::float_type, {3, 4, 5}};
    std::vector<float> x2_data = {
        0.19665868, 0.49490562, 0.73175228, 0.89251999, 0.08735652, 0.25944536, 0.37003717,
        0.09387889, 0.75490936, 0.81022481, 0.9987667,  0.04082882, 0.26160334, 0.85590193,
        0.80221833, 0.11203218, 0.31701572, 0.45973754, 0.3452479,  0.85151585, 0.86455042,
        0.19206577, 0.09922319, 0.58911914, 0.15871974, 0.61540675, 0.21682354, 0.69036427,
        0.77451157, 0.91950467, 0.52659111, 0.80857867, 0.63179264, 0.10085509, 0.96412482,
        0.42412458, 0.0330562,  0.13279482, 0.39372801, 0.80698385, 0.1182876,  0.75943908,
        0.59421519, 0.66827559, 0.09009574, 0.66649037, 0.43015355, 0.37795428, 0.11304274,
        0.37406792, 0.33043231, 0.32357327, 0.38079892, 0.42659918, 0.55308245, 0.49437723,
        0.95926415, 0.99762983, 0.70624046, 0.24298556};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.3195768,  0.92158614, 0.98164236, 1.20559466, 0.14507291, 0.71879884, 0.60203336,
        0.40083822, 0.73744823, 0.97361497, 1.04963956, 0.33451816, 0.5262512,  0.96263736,
        1.09464615, 0.46791396, 0.90542384, 1.05180592, 0.78995572, 0.90429304, 0.64010028,
        1.29062741, 1.31086115, 1.72652878, 0.23316878, 1.14509684, 0.85704442, 0.73375098,
        1.1197959,  1.48742487, 1.46556673, 0.67672563, 0.86988939, 1.26078125, 1.67521536,
        0.76174542, 1.26082452, 1.47107559, 1.17750291, 1.351588,   0.66717038, 0.57394148,
        0.72380011, 1.1455959,  0.17027018, 0.60247933, 0.46530117, 0.48794463, 1.10799312,
        1.24880054, 1.19090614, 0.50601796, 0.60271763, 0.82771923, 1.27385264, 0.35771131,
        0.33482015, 0.51852039, 0.5541507,  1.21648601};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_element_wise_multiplication_and_row_sum_test)
{
    migraphx::program p =
        migraphx::parse_onnx("einsum_element_wise_multiplication_and_row_sum_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {3}};
    std::vector<float> x1_data = {0.66866322, 0.01371844, 0.85036724};

    migraphx::shape x2_shape{migraphx::shape::float_type, {3, 4}};
    std::vector<float> x2_data = {0.72487469,
                                  0.24707426,
                                  0.8735483,
                                  0.04525622,
                                  0.52379655,
                                  0.32056461,
                                  0.51596208,
                                  0.10696902,
                                  0.08682559,
                                  0.95054461,
                                  0.16377484,
                                  0.61029108};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.2642773, 0.02012896, 1.54038595};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

// TODO /code/AMDMIGraphX/src/include/migraphx/op/dot.hpp:84: compute_shape: DOT: static outer
// dimensions of A and B mismatch: {1, 4, 2} x {2, 4, 1}
// TEST_CASE(einsum_3_inputs_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_3_inputs_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2}};
//     std::vector<float> x1_data = {0.78808491,
//                                   0.6661874,
//                                   0.4170594,
//                                   0.80972418,
//                                   0.22687053,
//                                   0.52144567,
//                                   0.70463225,
//                                   0.8934412};

//     migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2}};
//     std::vector<float> x2_data = {0.98518483, 0.61526655, 0.89011461, 0.02600793};

//     migraphx::shape x3_shape{migraphx::shape::float_type, {2, 2, 2}};
//     std::vector<float> x3_data = {0.04135729,
//                                   0.36723732,
//                                   0.82196749,
//                                   0.35332048,
//                                   0.92673273,
//                                   0.50014512,
//                                   0.91129541,
//                                   0.97557965};

//     migraphx::parameter_map pm;
//     pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
//     pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};
//     pm["x3"] = migraphx::argument{x3_shape, x3_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold = {1.54312876,
//                                0.59155446,
//                                1.19274407,
//                                0.56709538,
//                                2.79449706,
//                                1.61644006,
//                                2.15997517,
//                                1.5496049};
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }

TEST_CASE(einsum_bilinear_transformation_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_bilinear_transformation_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x1_data = {
        0.34096073, 0.38172764, 0.36543085, 0.28104558, 0.0556053, 0.23574725};

    migraphx::shape x2_shape{migraphx::shape::float_type, {5, 3, 7}};
    std::vector<float> x2_data = {
        0.27525548, 0.55922006, 0.28504873, 0.48681888, 0.7527785,  0.76094518, 0.99365312,
        0.76470274, 0.44406814, 0.24103473, 0.25141801, 0.51590554, 0.78834812, 0.96411404,
        0.01325493, 0.21739615, 0.25936655, 0.23025532, 0.85856546, 0.33609085, 0.33413049,
        0.60163776, 0.61253489, 0.84028869, 0.2593441,  0.53611056, 0.05595679, 0.30129639,
        0.44404875, 0.71431542, 0.95123376, 0.71387725, 0.05743836, 0.35266739, 0.53284905,
        0.07799213, 0.3639559,  0.72199632, 0.0920087,  0.71882463, 0.09804492, 0.79378518,
        0.2149909,  0.62017677, 0.57284093, 0.1480283,  0.65038853, 0.47830376, 0.18202239,
        0.37421293, 0.65768777, 0.2465394,  0.80183419, 0.65855262, 0.40956847, 0.36430994,
        0.4464513,  0.65720017, 0.29603235, 0.21994904, 0.31797431, 0.64774027, 0.71807814,
        0.67456442, 0.37665375, 0.84645173, 0.10965697, 0.57469259, 0.68129292, 0.28780513,
        0.50772577, 0.67820423, 0.92720621, 0.52615601, 0.5507361,  0.55419857, 0.37244191,
        0.52378246, 0.29057448, 0.14684616, 0.60456568, 0.79814119, 0.51783395, 0.69921548,
        0.12310853, 0.18934048, 0.98081268, 0.51493817, 0.1279986,  0.3868668,  0.42396674,
        0.04160038, 0.56299233, 0.40414454, 0.73163413, 0.3126024,  0.75276068, 0.88847181,
        0.96703089, 0.34357903, 0.34495332, 0.73431682, 0.01318382, 0.15232141, 0.88949811};

    migraphx::shape x3_shape{migraphx::shape::float_type, {2, 7}};
    std::vector<float> x3_data = {0.22897831,
                                  0.68897913,
                                  0.55615068,
                                  0.77395085,
                                  0.44879247,
                                  0.42608676,
                                  0.45303661,
                                  0.04397996,
                                  0.44780993,
                                  0.98314993,
                                  0.32980751,
                                  0.57814391,
                                  0.91010863,
                                  0.53235916};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};
    pm["x3"] = migraphx::argument{x3_shape, x3_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.82915577,
                               1.88971744,
                               1.84172272,
                               2.0310065,
                               1.91888787,
                               1.11119172,
                               1.03903856,
                               1.03828167,
                               1.17052253,
                               0.98080627};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_common_1_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x1_data = {0.35498396,
                                  0.92145607,
                                  0.81807284,
                                  0.37990484,
                                  0.22314499,
                                  0.90337144,
                                  0.02492543,
                                  0.36666091,
                                  0.33262049,
                                  0.37052745,
                                  0.01950226,
                                  0.83690205,
                                  0.61551503,
                                  0.55244304,
                                  0.62696715,
                                  0.74933671};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x2_data = {0.44903857,
                                  0.47304138,
                                  0.63679145,
                                  0.78101353,
                                  0.41525864,
                                  0.57356733,
                                  0.83636479,
                                  0.01236986,
                                  0.10068789,
                                  0.46623025,
                                  0.29825429,
                                  0.56816588,
                                  0.00558546,
                                  0.91900877,
                                  0.74972012,
                                  0.4509882};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.59528833,
                               0.52753278,
                               0.67592725,
                               0.61080723,
                               0.81765261,
                               0.30223943,
                               0.68890669,
                               0.0253823,
                               0.20624196,
                               0.31954056,
                               0.34237582,
                               0.51113793,
                               0.48131582,
                               0.6127432,
                               0.39205418,
                               0.8079919};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_common_2_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x1_data = {0.77858647,
                                  0.8659616,
                                  0.89981848,
                                  0.45454779,
                                  0.27364842,
                                  0.69225887,
                                  0.01304595,
                                  0.14404551,
                                  0.47394644,
                                  0.39058325,
                                  0.977306,
                                  0.90298946,
                                  0.01456065,
                                  0.70478062,
                                  0.92796867,
                                  0.00407166};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x2_data = {0.12299003,
                                  0.42677007,
                                  0.84213152,
                                  0.26884624,
                                  0.85685616,
                                  0.53033816,
                                  0.61543941,
                                  0.00586418,
                                  0.79310638,
                                  0.66468861,
                                  0.22797244,
                                  0.32789713,
                                  0.01537162,
                                  0.28328088,
                                  0.39257709,
                                  0.83954883};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {2.51890769,
                               1.78883817,
                               2.11484282,
                               1.38804189,
                               2.81881969,
                               1.09537142,
                               3.0398521,
                               1.07377846};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_common_3_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_3_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x1_data = {0.22151958,
                                  0.19284961,
                                  0.8126814,
                                  0.02360209,
                                  0.99137254,
                                  0.0550951,
                                  0.34794661,
                                  0.03083101,
                                  0.03127261,
                                  0.04609321,
                                  0.02422953,
                                  0.30878066,
                                  0.42532866,
                                  0.02191982,
                                  0.34276933,
                                  0.66997637};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x2_data = {0.76051399,
                                  0.92365044,
                                  0.14703117,
                                  0.07201171,
                                  0.81879942,
                                  0.91050362,
                                  0.90936259,
                                  0.94197062,
                                  0.73971579,
                                  0.08809791,
                                  0.17392649,
                                  0.36623704,
                                  0.23731799,
                                  0.67476051,
                                  0.97480632,
                                  0.35175013};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.62099637,
                               2.20329706,
                               0.6457657,
                               1.61829179,
                               0.4142793,
                               0.52881853,
                               2.00689201,
                               2.20807455};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_common_4_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_4_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 3, 2}};
    std::vector<float> x1_data = {0.56144416, 0.70795103, 0.10800643, 0.85461707, 0.53053745,
                                  0.42957473, 0.2801385,  0.91878799, 0.51160639, 0.90354742,
                                  0.83131358, 0.84237736, 0.01078178, 0.75952001, 0.74426499,
                                  0.70506648, 0.65528756, 0.54674358, 0.3923791,  0.33558121,
                                  0.18089114, 0.41982192, 0.50568299, 0.83929267};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 4, 2}};
    std::vector<float> x2_data = {
        0.71114916, 0.10373848, 0.85011488, 0.08836512, 0.01426097, 0.63389153, 0.3714056,
        0.42466907, 0.5412509,  0.12682203, 0.88595126, 0.09839624, 0.10689487, 0.1196194,
        0.5887543,  0.51683836, 0.50278953, 0.94187525, 0.98227159, 0.57961915, 0.12739494,
        0.59140361, 0.34997506, 0.43158845, 0.60170823, 0.06098434, 0.24573198, 0.15357368,
        0.99864135, 0.92721276, 0.81457582, 0.49836327};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.4727123,  0.53985021, 0.4567709,  0.50916841, 0.16546536, 0.16733621, 0.5432748,
        0.40304363, 0.42185469, 0.48897721, 0.27986976, 0.37947168, 0.26814778, 0.33859434,
        0.13985024, 0.63979763, 0.39149714, 0.54216399, 0.1627699,  0.76819843, 0.55678123,
        0.81939007, 0.18962783, 0.92481237, 0.72079407, 0.45082298, 0.45055642, 0.33157342,
        1.03829331, 1.13974038, 0.51179445, 0.56477273, 0.84443597, 0.9605734,  0.40682645,
        0.46530252, 0.25656293, 0.14795654, 0.70300118, 0.48686388, 0.13444625, 0.10892434,
        0.56990961, 0.35657337, 0.35545733, 0.25315575, 1.28319881, 0.83018978};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
