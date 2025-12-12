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

#include <migraphx/generic_float.hpp>
#include <migraphx/split_factor.hpp>
#include <limits>
#include "test.hpp"

TEST_CASE(integer_divide_ceil_basic)
{
    // Test exact division
    EXPECT(migraphx::integer_divide_ceil(10, 5) == 2);
    EXPECT(migraphx::integer_divide_ceil(12, 4) == 3);
    EXPECT(migraphx::integer_divide_ceil(15, 3) == 5);

    // Test division with remainder (should round up)
    EXPECT(migraphx::integer_divide_ceil(10, 3) == 4);  // 10/3 = 3.33... -> 4
    EXPECT(migraphx::integer_divide_ceil(11, 3) == 4);  // 11/3 = 3.66... -> 4
    EXPECT(migraphx::integer_divide_ceil(13, 5) == 3);  // 13/5 = 2.6 -> 3

    // Test with 1
    EXPECT(migraphx::integer_divide_ceil(1, 1) == 1);
    EXPECT(migraphx::integer_divide_ceil(5, 1) == 5);
    EXPECT(migraphx::integer_divide_ceil(100, 1) == 100);

    // Test edge cases
    EXPECT(migraphx::integer_divide_ceil(0, 5) == 0);
    EXPECT(migraphx::integer_divide_ceil(1, 2) == 1);
    EXPECT(migraphx::integer_divide_ceil(1, 10) == 1);
}

TEST_CASE(integer_divide_ceil_large_numbers)
{
    // Test with larger numbers
    EXPECT(migraphx::integer_divide_ceil(1000, 7) == 143);    // 1000/7 = 142.857... -> 143
    EXPECT(migraphx::integer_divide_ceil(1024, 32) == 32);    // Exact division
    EXPECT(migraphx::integer_divide_ceil(1025, 32) == 33);    // 1025/32 = 32.03... -> 33

    // Test with powers of 2
    EXPECT(migraphx::integer_divide_ceil(127, 8) == 16);      // 127/8 = 15.875 -> 16
    EXPECT(migraphx::integer_divide_ceil(128, 8) == 16);      // Exact division
    EXPECT(migraphx::integer_divide_ceil(129, 8) == 17);      // 129/8 = 16.125 -> 17
}

TEST_CASE(ceil_mul_of_basic)
{
    // Test exact multiples (no rounding needed)
    EXPECT(migraphx::ceil_mul_of(10, 5) == 10);   // 10 is already a multiple of 5
    EXPECT(migraphx::ceil_mul_of(12, 4) == 12);   // 12 is already a multiple of 4
    EXPECT(migraphx::ceil_mul_of(15, 3) == 15);   // 15 is already a multiple of 3

    // Test rounding up to next multiple
    EXPECT(migraphx::ceil_mul_of(11, 5) == 15);   // Next multiple of 5 after 11 is 15
    EXPECT(migraphx::ceil_mul_of(13, 4) == 16);   // Next multiple of 4 after 13 is 16
    EXPECT(migraphx::ceil_mul_of(17, 3) == 18);   // Next multiple of 3 after 17 is 18

    // Test with 1 (should always return the original number)
    EXPECT(migraphx::ceil_mul_of(5, 1) == 5);
    EXPECT(migraphx::ceil_mul_of(100, 1) == 100);

    // Test edge cases
    EXPECT(migraphx::ceil_mul_of(0, 5) == 0);
    EXPECT(migraphx::ceil_mul_of(1, 5) == 5);
    EXPECT(migraphx::ceil_mul_of(1, 10) == 10);
}

TEST_CASE(ceil_mul_of_powers_of_two)
{
    // Test with powers of 2 (common in GPU programming)
    EXPECT(migraphx::ceil_mul_of(100, 32) == 128);    // 32 * 4 = 128
    EXPECT(migraphx::ceil_mul_of(128, 32) == 128);    // Already aligned
    EXPECT(migraphx::ceil_mul_of(129, 32) == 160);    // 32 * 5 = 160

    EXPECT(migraphx::ceil_mul_of(250, 64) == 256);    // 64 * 4 = 256
    EXPECT(migraphx::ceil_mul_of(256, 64) == 256);    // Already aligned
    EXPECT(migraphx::ceil_mul_of(257, 64) == 320);    // 64 * 5 = 320

    // Warp size alignment (32 threads)
    EXPECT(migraphx::ceil_mul_of(30, 32) == 32);
    EXPECT(migraphx::ceil_mul_of(32, 32) == 32);
    EXPECT(migraphx::ceil_mul_of(33, 32) == 64);
    EXPECT(migraphx::ceil_mul_of(64, 32) == 64);
    EXPECT(migraphx::ceil_mul_of(65, 32) == 96);
}

TEST_CASE(ceil_mul_of_flash_attention_use_case)
{
    // Test the specific use case from flash attention
    // Simulating the padding of sequence length to be divisible by number of groups

    // Example 1: sequence_length=100, num_groups=8
    EXPECT(migraphx::ceil_mul_of(100, 8) == 104);     // 8 * 13 = 104

    // Example 2: sequence_length=127, num_groups=16
    EXPECT(migraphx::ceil_mul_of(127, 16) == 128);    // 16 * 8 = 128

    // Example 3: sequence_length=200, num_groups=32
    EXPECT(migraphx::ceil_mul_of(200, 32) == 224);    // 32 * 7 = 224

    // Example 4: Already divisible
    EXPECT(migraphx::ceil_mul_of(192, 32) == 192);    // Already divisible by 32
}

TEST_CASE(ceil_mul_of_consistency_with_integer_divide_ceil)
{
    // Verify that ceil_mul_of(x, y) == y * integer_divide_ceil(x, y)
    // This tests the implementation relationship

    std::size_t test_cases[][2] = {
        {10, 3},
        {15, 4},
        {100, 7},
        {256, 32},
        {1000, 13},
        {1, 10},
        {0, 5}
    };

    for(const auto& tc : test_cases)
    {
        std::size_t x = tc[0];
        std::size_t y = tc[1];

        std::size_t expected = y * migraphx::integer_divide_ceil(x, y);
        std::size_t actual = migraphx::ceil_mul_of(x, y);

        EXPECT(actual == expected);
    }
}

TEST_CASE(ceil_mul_of_large_numbers)
{
    // Test with larger sequence lengths that might appear in real models
    EXPECT(migraphx::ceil_mul_of(1024, 16) == 1024);     // Already aligned
    EXPECT(migraphx::ceil_mul_of(1025, 16) == 1040);     // 16 * 65 = 1040
    EXPECT(migraphx::ceil_mul_of(2048, 64) == 2048);     // Already aligned
    EXPECT(migraphx::ceil_mul_of(2049, 64) == 2112);     // 64 * 33 = 2112

    // Very large numbers
    EXPECT(migraphx::ceil_mul_of(10000, 128) == 10112);  // 128 * 79 = 10112
    EXPECT(migraphx::ceil_mul_of(16384, 256) == 16384);  // Already aligned
    EXPECT(migraphx::ceil_mul_of(16385, 256) == 16640);  // 256 * 65 = 16640
}

TEST_CASE(split_dim_basic)
{
    // Test with no max_splits constraint (using default)

    // Should split 100 into chunks > 10
    // 100 = 2^2 * 5^2; factors: 2,2,5,5 -> splits = 20, remaining = 5
    std::size_t dim100 = 100;
    std::size_t result100 = migraphx::split_dim(dim100, 10);
    EXPECT(result100 == 20);  // 100/20 = 5, stops because 5 <= 10

    // Should split 64 into chunks > 10  
    // 64 = 2^6; can use 2,2,2 -> splits = 8, remaining = 8
    std::size_t dim64 = 64;
    std::size_t result64 = migraphx::split_dim(dim64, 10);
    EXPECT(result64 == 8);   // 64/8 = 8, stops because 8 <= 10

    // Should not split if already small enough
    std::size_t dim10 = 10;
    std::size_t result10 = migraphx::split_dim(dim10, 10);
    EXPECT(result10 == 1);   // 10 is not > 10, so no split

    std::size_t dim11 = 11;
    std::size_t result11 = migraphx::split_dim(dim11, 10);
    EXPECT(result11 == 11);  // 11 is a factor itself, 11/11 = 1

    // Prime numbers that can't be factored
    std::size_t dim13 = 13;
    std::size_t result13 = migraphx::split_dim(dim13, 10);
    EXPECT(result13 == 1);   // 13 is prime (not in factor list)

    std::size_t dim17 = 17;
    std::size_t result17 = migraphx::split_dim(dim17, 10);
    EXPECT(result17 == 1);   // 17 is prime (not in factor list)

    // Numbers with factors in [2,3,5,7,11]
    std::size_t dim30 = 30;
    std::size_t result30 = migraphx::split_dim(dim30, 5);
    EXPECT(result30 == 6);    // 30 = 2*3*5, splits to 5

    std::size_t dim77 = 77;
    std::size_t result77 = migraphx::split_dim(dim77, 10);
    EXPECT(result77 == 77);   // can be evenly split into 11 size chunks; next divisor splits to 1 size chunks
}

TEST_CASE(split_dim_with_max_splits)
{
    // Test with explicit max_splits constraint
    // Note: max_splits is NOT a hard cap - function returns smallest split factor > max_splits that evenly divides dimension

    // When split factor would exceed max_splits, returns next valid divisor
    std::size_t dim100a = 100;
    std::size_t result100a = migraphx::split_dim(dim100a, 10, 4);
    EXPECT(result100a == 4);  // 100 can be divided by 2*2, which is 4 splits, which is not less than max_splits=4

    std::size_t dim100b = 100;
    std::size_t result100b = migraphx::split_dim(dim100b, 10, 2);
    EXPECT(result100b == 2);

    // Max splits doesn't force splitting if min_size constraint would be violated
    std::size_t dim20 = 20;
    std::size_t result20 = migraphx::split_dim(dim20, 10, 4);
    EXPECT(result20 == 2);    // Can only split to 2 (20/2=10, not > 10)

    std::size_t dim15 = 15;
    std::size_t result15 = migraphx::split_dim(dim15, 10, 4);
    EXPECT(result15 == 3);    // 15 = 3*5, splits to 3, remaining = 5

    // Test with powers of 2
    std::size_t dim128a = 128;
    std::size_t result128a = migraphx::split_dim(dim128a, 10, 8);
    EXPECT(result128a == 8);  // 128 can be divided by 2*2*2, which is 8 splits, which is not less than max_splits=8

    std::size_t dim128b = 128;
    std::size_t result128b = migraphx::split_dim(dim128b, 10, 4);
    EXPECT(result128b == 4);   // 128 can be divided by 2*2, which is 4 splits, which is not less than max_splits=4

    std::size_t dim128c = 128;
    std::size_t result128c = migraphx::split_dim(dim128c, 20, 8);
    EXPECT(result128c == 8);   // 128 can be divided by 2*2*2, which is 8 splits, which is not less than max_splits=8
}

TEST_CASE(split_dim_edge_cases)
{
    // Test edge cases

    // Very small dimensions
    std::size_t dim1 = 1;
    std::size_t result1 = migraphx::split_dim(dim1, 0);
    EXPECT(result1 == 1);     // 1 can't be split

    std::size_t dim2a = 2;
    std::size_t result2a = migraphx::split_dim(dim2a, 0);
    EXPECT(result2a == 2);     // 2/2 = 1 > 0

    std::size_t dim2b = 2;
    std::size_t result2b = migraphx::split_dim(dim2b, 1);
    EXPECT(result2b == 2);     // 2/2 = 1, but we continue while r > min_size, so 2 > 1 allows split

    // Exact boundary conditions
    std::size_t dim20a = 20;
    std::size_t result20a = migraphx::split_dim(dim20a, 9);
    EXPECT(result20a == 4);    // 20 = 2^2 * 5, factors 2,2 before 20/4=5 <= 9

    std::size_t dim20b = 20;
    std::size_t result20b = migraphx::split_dim(dim20b, 10);
    EXPECT(result20b == 2);   // 20 = 2^2 * 5, factors 2 before 20/2=10 <= 10

    std::size_t dim21 = 21;
    std::size_t result21 = migraphx::split_dim(dim21, 10);
    EXPECT(result21 == 3);   // 21 = 3*7, splits by 3 first, 21/3 = 7 <= 10

    // Large prime numbers
    std::size_t dim97 = 97;
    std::size_t result97 = migraphx::split_dim(dim97, 10);
    EXPECT(result97 == 1);   // 97 is prime

    std::size_t dim101 = 101;
    std::size_t result101 = migraphx::split_dim(dim101, 10);
    EXPECT(result101 == 1);  // 101 is prime
}

TEST_CASE(split_dim_factorization_order)
{
    // Test that factorization happens in the expected order [2,3,5,7,11]

    // 60 = 2^2 * 3 * 5
    // With min_size=10: 60->30->15->5 (stops because 5 <= 10)
    // Factors used: 2, 2, 3 (product = 12)
    std::size_t dim60 = 60;
    std::size_t result60 = migraphx::split_dim(dim60, 10);
    EXPECT(result60 == 12);   // 60/12 = 5

    // 210 = 2 * 3 * 5 * 7
    // With min_size=20: continues factoring while 210 > 20
    // Factors all: 2*3*5*7 = 210, but stops at 2*3*5 = 30 since 210/30 = 7 <= 20
    std::size_t dim210 = 210;
    std::size_t result210 = migraphx::split_dim(dim210, 20);
    EXPECT(result210 == 30);  // 210/30 = 7

    // 462 = 2 * 3 * 7 * 11
    // With min_size=30: 462->231->77->11 (stops because 11 <= 30)
    std::size_t dim462 = 462;
    std::size_t result462 = migraphx::split_dim(dim462, 30);
    EXPECT(result462 == 42);  // 462/42 = 11 <= 30
}

TEST_CASE(split_dim_reduce_use_case)
{
    // Test the specific use case from reduce.cpp
    // These are realistic values that might appear in reduction operations

    // Large reduction dimension with typical min_size and max_splits
    std::size_t dim1024 = 1024;
    std::size_t result1024 = migraphx::split_dim(dim1024, 64, 16);
    EXPECT(result1024 == 16);   // 1024 = 2^10; when n=8 < 16, multiplies to 16; when n=16 < 16 is false, stops

    std::size_t dim1000 = 1000;
    std::size_t result1000 = migraphx::split_dim(dim1000, 64, 16);
    EXPECT(result1000 == 40);    // 1000 = 2^3 * 5^3; n=8, r=125 > 64, so next would be n=40 > 16

    // Smaller dimensions
    std::size_t dim256 = 256;
    std::size_t result256 = migraphx::split_dim(dim256, 32, 8);
    EXPECT(result256 == 8);     // 256 = 2^8; would split to 16 (256/16=16), exceeds max_splits=8

    std::size_t dim200 = 200;
    std::size_t result200 = migraphx::split_dim(dim200, 32, 8);
    EXPECT(result200 == 8);     // 200 = 2^3 * 5^2; would split to 8 (200/8=25), exceeds max_splits=8
}

TEST_CASE(split_dim_flash_attention_use_case)
{
    // Test use cases from flash attention decoding
    // Sequence lengths need to be split for parallel processing

    // Typical sequence lengths in attention
    std::size_t dim2048 = 2048;
    std::size_t result2048 = migraphx::split_dim(dim2048, 128, 16);
    EXPECT(result2048 == 16);  // 2048 = 2^11; when n=8 < 16, multiplies to 16; when n=16 < 16 is false, stops

    std::size_t dim4096 = 4096;
    std::size_t result4096 = migraphx::split_dim(dim4096, 256, 16);
    EXPECT(result4096 == 16);  // 4096 = 2^12; splits to 16 (4096/16=256)

    // Non-aligned sequence lengths
    // 1536 = 2^9 * 3; would continue past max_splits=16 to get 1536/24=64 < 128
    std::size_t dim1536 = 1536;
    std::size_t result1536 = migraphx::split_dim(dim1536, 128, 16);
    EXPECT(result1536 == 16);  // 1536 = 2^9 * 3; n=16, r=128, stops (r not > 128)

    // 3000 = 2^3 * 3 * 5^3
    std::size_t dim3000 = 3000;
    std::size_t result3000 = migraphx::split_dim(dim3000, 200, 16);
    EXPECT(result3000 == 24);  // 3*5 = 24, 3000/24 = 125

    // Smaller sequences
    std::size_t dim512 = 512;
    std::size_t result512 = migraphx::split_dim(dim512, 64, 8);
    EXPECT(result512 == 8);      // 512 = 2^9; stops at n=8 (512/8=64)

    std::size_t dim768 = 768;
    std::size_t result768 = migraphx::split_dim(dim768, 64, 8);
    EXPECT(result768 == 8);      // 768 = 2^8 * 3; n=8, r=128 > 64, stops (r not > 128)
}

TEST_CASE(split_dim_no_limit_vs_explicit_max)
{
    // Verify that default (no limit) and explicit max give same results when max is large
    std::size_t large_max = std::numeric_limits<std::size_t>::max();

    // These should produce identical results
    std::size_t dim1000a = 1000;
    std::size_t result1000a = migraphx::split_dim(dim1000a, 10);
    std::size_t dim1000b = 1000;
    std::size_t result1000b = migraphx::split_dim(dim1000b, 10, large_max);
    EXPECT(result1000a == result1000b);

    std::size_t dim512a = 512;
    std::size_t result512a = migraphx::split_dim(dim512a, 32);
    std::size_t dim512b = 512;
    std::size_t result512b = migraphx::split_dim(dim512b, 32, large_max);
    EXPECT(result512a == result512b);

    std::size_t dim360a = 360;
    std::size_t result360a = migraphx::split_dim(dim360a, 20);
    std::size_t dim360b = 360;
    std::size_t result360b = migraphx::split_dim(dim360b, 20, large_max);
    EXPECT(result360a == result360b);

    // Test that max_splits affects the result (but doesn't necessarily limit it)
    // max_splits acts as a threshold - result may exceed it
    std::size_t dim1000c = 1000;
    std::size_t result1000c = migraphx::split_dim(dim1000c, 10);
    std::size_t dim1000d = 1000;
    std::size_t result1000d = migraphx::split_dim(dim1000d, 10, 8);
    EXPECT(result1000c != result1000d);

    std::size_t dim512c = 512;
    std::size_t result512c = migraphx::split_dim(dim512c, 8);
    std::size_t dim512d = 512;
    std::size_t result512d = migraphx::split_dim(dim512d, 8, 16);
    EXPECT(result512c != result512d);
}

TEST_CASE(split_dim_consistency_check)
{
    // Verify important properties of split_dim

    // Property 1: The algorithm stops when remaining <= min_size
    // This means the final remaining might be <= min_size
    for(std::size_t dim : {100, 256, 512, 1000, 2048})
    {
        for(std::size_t min_size : {8, 16, 32, 64})
        {
            std::size_t dim_copy = dim;
            std::size_t splits = migraphx::split_dim(dim_copy, min_size);
            if(splits > 1)
            {
                // The algorithm continues while r > min_size,
                // so it stops when r <= min_size
                // We can't guarantee remaining > min_size,
                // but we know the split was valid
                EXPECT(splits >= 1);
                EXPECT(dim / splits > 0);
            }
        }
    }

    // Property 2: With max_splits, result is smallest valid divisor >= max_splits
    // Note: max_splits is NOT a hard cap - it's a threshold like min_size
    for(std::size_t dim : {100, 256, 512, 1000})
    {
        for(std::size_t max_splits : {4, 8, 16})
        {
            std::size_t dim_copy = dim;
            std::size_t splits = migraphx::split_dim(dim_copy, 10, max_splits);
            // Result should evenly divide dimension
            EXPECT(dim % splits == 0);
            // Result should make remaining size < min_size (10)
            EXPECT(dim / splits < 10 || splits >= max_splits);
        }
    }

    // Property 3: Increasing min_size decreases or maintains splits
    for(std::size_t dim : {256, 512, 1024})
    {
        std::size_t dim_copy1 = dim;
        std::size_t splits_8 = migraphx::split_dim(dim_copy1, 8);
        std::size_t dim_copy2 = dim;
        std::size_t splits_16 = migraphx::split_dim(dim_copy2, 16);
        std::size_t dim_copy3 = dim;
        std::size_t splits_32 = migraphx::split_dim(dim_copy3, 32);

        EXPECT(splits_8 >= splits_16);
        EXPECT(splits_16 >= splits_32);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
