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
#include <migraphx/kernels/matmul.hpp>
#include <migraphx/kernels/test.hpp>
#include <migraphx/kernels/float_equal.hpp>

// matmul is wave-collective: run every case on one whole wave
TEST_LAUNCH(MIGRAPHX_WAVEFRONTSIZE, MIGRAPHX_WAVEFRONTSIZE)

// Operands from a formula so every lane generates them: small integers that
// are exact in half and bf16 and whose sums stay exact in float
template <class T>
__device__ T a_value(migraphx::index_int i, migraphx::index_int k)
{
    T x = (i * 7 + k * 3) % 5;
    return x - T{2};
}

template <class T>
__device__ T b_value(migraphx::index_int k, migraphx::index_int j)
{
    T x = (k * 5 + j * 11) % 7;
    return x - T{3};
}

// The wave multiplies through the loaders into LDS, then every lane checks
// the whole product so the failure is seen from lane 0
template <class T, migraphx::index_int M, migraphx::index_int N, migraphx::index_int K>
__device__ bool matmul_matches()
{
    __shared__ float c[M * N];
    migraphx::matmul<T, M, N, K>([](auto i, auto k) { return a_value<T>(i, k); },
                                 [](auto k, auto j) { return b_value<T>(k, j); },
                                 [&](auto x, auto i, auto j) { c[i * N + j] = x; });
    __syncthreads();
    bool ok = true;
    for(migraphx::index_int i = 0; i < M; i++)
    {
        for(migraphx::index_int j = 0; j < N; j++)
        {
            float ref = 0;
            for(migraphx::index_int k = 0; k < K; k++)
                ref += a_value<float>(i, k) * b_value<float>(k, j);
            if(not migraphx::float_equal(c[i * N + j], ref))
                ok = false;
        }
    }
    __syncthreads();
    return ok;
}

// Shapes the matrix instructions cover, over several k steps and blocks
template <class T>
TEST_CASE_TEMPLATE(matmul_blocks)
{
    EXPECT(matmul_matches<T, 16, 16, 16>());
    EXPECT(matmul_matches<T, 16, 16, 128>());
    EXPECT(matmul_matches<T, 32, 32, 48>());
}
TEST_CASE_REGISTER(matmul_blocks<migraphx::half>)
TEST_CASE_REGISTER(matmul_blocks<migraphx::bf16>)
TEST_CASE_REGISTER(matmul_blocks<float>)

// Shapes no matrix instruction covers take the per-lane path
template <class T>
TEST_CASE_TEMPLATE(matmul_odd_shapes)
{
    EXPECT(matmul_matches<T, 8, 12, 5>());
    EXPECT(matmul_matches<T, 7, 3, 9>());
    EXPECT(matmul_matches<T, 16, 18, 16>());
}
TEST_CASE_REGISTER(matmul_odd_shapes<migraphx::half>)
TEST_CASE_REGISTER(matmul_odd_shapes<float>)
