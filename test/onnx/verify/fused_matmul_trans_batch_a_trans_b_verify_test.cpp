/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// End-to-end check of the rank-4 transBatchA + transB lowering, which (unlike rank-3)
// distinguishes ORT's transBatch permutation from the alternate one. ORT's
// MatMulComputeHelper permutes a rank-4 A with [1, 2, 0, 3] (not [2, 0, 1, 3]), so the gold
// values below are only reproduced if the parser emits that permutation.
TEST_CASE(fused_matmul_trans_batch_a_trans_b_verify_test)
{
    migraphx::program p = read_onnx("fused_matmul_trans_batch_a_trans_b_test.onnx");
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape a_shape{input_type, {2, 3, 4, 7}};
    migraphx::shape b_shape{input_type, {3, 4, 8, 7}};

    std::vector<float> a_data(2 * 3 * 4 * 7);
    std::iota(a_data.begin(), a_data.end(), 0); // 0..167
    std::vector<float> b_data(3 * 4 * 8 * 7);
    std::iota(b_data.begin(), b_data.end(), 0); // 0..671

    migraphx::parameter_map pp;
    pp["1"] = migraphx::argument(a_shape, a_data.data());
    pp["2"] = migraphx::argument(b_shape, b_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold values generated with numpy:
    // >>> A = np.arange(168, dtype=np.float32).reshape(2, 3, 4, 7)
    // >>> B = np.arange(672, dtype=np.float32).reshape(3, 4, 8, 7)
    // >>> Ap = np.transpose(A, (1, 2, 0, 3))   # transBatchA -> [3, 4, 2, 7]
    // >>> Bp = np.transpose(B, (0, 1, 3, 2))   # transB      -> [3, 4, 7, 8]
    // >>> Y = Ap @ Bp                          # -> [3, 4, 2, 8]
    std::vector<float> gold = {
        91.0f,     238.0f,    385.0f,    532.0f,    679.0f,    826.0f,    973.0f,    1120.0f,
        1855.0f,   6118.0f,   10381.0f,  14644.0f,  18907.0f,  23170.0f,  27433.0f,  31696.0f,
        4158.0f,   4648.0f,   5138.0f,   5628.0f,   6118.0f,   6608.0f,   7098.0f,   7588.0f,
        38850.0f,  43456.0f,  48062.0f,  52668.0f,  57274.0f,  61880.0f,  66486.0f,  71092.0f,
        13713.0f,  14546.0f,  15379.0f,  16212.0f,  17045.0f,  17878.0f,  18711.0f,  19544.0f,
        81333.0f,  86282.0f,  91231.0f,  96180.0f,  101129.0f, 106078.0f, 111027.0f, 115976.0f,
        28756.0f,  29932.0f,  31108.0f,  32284.0f,  33460.0f,  34636.0f,  35812.0f,  36988.0f,
        129304.0f, 134596.0f, 139888.0f, 145180.0f, 150472.0f, 155764.0f, 161056.0f, 166348.0f,
        49287.0f,  50806.0f,  52325.0f,  53844.0f,  55363.0f,  56882.0f,  58401.0f,  59920.0f,
        182763.0f, 188398.0f, 194033.0f, 199668.0f, 205303.0f, 210938.0f, 216573.0f, 222208.0f,
        75306.0f,  77168.0f,  79030.0f,  80892.0f,  82754.0f,  84616.0f,  86478.0f,  88340.0f,
        241710.0f, 247688.0f, 253666.0f, 259644.0f, 265622.0f, 271600.0f, 277578.0f, 283556.0f,
        106813.0f, 109018.0f, 111223.0f, 113428.0f, 115633.0f, 117838.0f, 120043.0f, 122248.0f,
        306145.0f, 312466.0f, 318787.0f, 325108.0f, 331429.0f, 337750.0f, 344071.0f, 350392.0f,
        143808.0f, 146356.0f, 148904.0f, 151452.0f, 154000.0f, 156548.0f, 159096.0f, 161644.0f,
        376068.0f, 382732.0f, 389396.0f, 396060.0f, 402724.0f, 409388.0f, 416052.0f, 422716.0f,
        186291.0f, 189182.0f, 192073.0f, 194964.0f, 197855.0f, 200746.0f, 203637.0f, 206528.0f,
        451479.0f, 458486.0f, 465493.0f, 472500.0f, 479507.0f, 486514.0f, 493521.0f, 500528.0f,
        234262.0f, 237496.0f, 240730.0f, 243964.0f, 247198.0f, 250432.0f, 253666.0f, 256900.0f,
        532378.0f, 539728.0f, 547078.0f, 554428.0f, 561778.0f, 569128.0f, 576478.0f, 583828.0f,
        287721.0f, 291298.0f, 294875.0f, 298452.0f, 302029.0f, 305606.0f, 309183.0f, 312760.0f,
        618765.0f, 626458.0f, 634151.0f, 641844.0f, 649537.0f, 657230.0f, 664923.0f, 672616.0f,
        346668.0f, 350588.0f, 354508.0f, 358428.0f, 362348.0f, 366268.0f, 370188.0f, 374108.0f,
        710640.0f, 718676.0f, 726712.0f, 734748.0f, 742784.0f, 750820.0f, 758856.0f, 766892.0f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
