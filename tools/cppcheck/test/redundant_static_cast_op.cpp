/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
// Test for RedundantStaticCastOp check

#include <cstddef>
#include <cstdint>

// Positive cases: the other operand already has the cast's exact integral type,
// so the usual arithmetic conversions of the binary operator already produce
// that type and the explicit cast is redundant. Each operator that applies the
// usual arithmetic conversions is covered, across both signednesses.

void test_unsigned_lt(std::uint64_t u, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    bool a = u < static_cast<std::uint64_t>(x);
    (void)a;
}

void test_signed_le(std::int64_t s, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    bool a = s <= static_cast<std::int64_t>(x);
    (void)a;
}

void test_unsigned_gt(unsigned u, short x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    bool a = u > static_cast<unsigned>(x);
    (void)a;
}

void test_signed_ge_reversed(std::int64_t s, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    bool a = static_cast<std::int64_t>(x) >= s;
    (void)a;
}

void test_unsigned_eq_size_t(std::size_t n, int i)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    bool a = n == static_cast<std::size_t>(i);
    (void)a;
}

void test_signed_ne_reversed(std::int32_t s, short x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    bool a = static_cast<std::int32_t>(x) != s;
    (void)a;
}

std::uint64_t test_unsigned_add(std::uint64_t u, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return u + static_cast<std::uint64_t>(x);
}

std::int64_t test_signed_sub_reversed(std::int64_t s, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return static_cast<std::int64_t>(x) - s;
}

std::uint64_t test_unsigned_mul(std::uint64_t u, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return u * static_cast<std::uint64_t>(x);
}

std::int64_t test_signed_div(std::int64_t s, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return s / static_cast<std::int64_t>(x);
}

std::uint64_t test_unsigned_mod(std::uint64_t u, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return u % static_cast<std::uint64_t>(x);
}

std::int64_t test_signed_and(std::int64_t s, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return s & static_cast<std::int64_t>(x);
}

std::uint64_t test_unsigned_or(std::uint64_t u, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return u | static_cast<std::uint64_t>(x);
}

std::int64_t test_signed_xor_reversed(std::int64_t s, int x)
{
    // cppcheck-suppress migraphx-RedundantStaticCastOp
    return static_cast<std::int64_t>(x) ^ s;
}

// Negative cases: nothing should be reported below.

// The cast's signedness differs from the other operand, so the cast changes the
// comparison.
void test_unsigned_operand_signed_cast(std::uint64_t u, int x)
{
    bool a = u <= static_cast<std::int64_t>(x);
    (void)a;
}

void test_signed_operand_unsigned_cast(std::int64_t s, int x)
{
    bool a = s <= static_cast<std::uint64_t>(x);
    (void)a;
}

// The operand is a different (narrower) type than the cast target, so the cast
// changes how it is promoted.
void test_different_width_unsigned(std::uint32_t u, int x)
{
    bool a = u <= static_cast<std::uint64_t>(x);
    (void)a;
}

void test_different_width_signed(int s, int x)
{
    bool a = s <= static_cast<std::int64_t>(x);
    (void)a;
}

// The cast source is unsigned with the same rank as the signed target, so the
// usual arithmetic conversions compare as unsigned, not signed: not redundant.
void test_unsigned_source_signed_target(int s, unsigned x)
{
    bool a = s == static_cast<int>(x);
    (void)a;
}

// Both operands are explicitly cast, removing either cast changes the operation.
void test_both_cast(int x, int y)
{
    bool a = static_cast<std::uint64_t>(x) <= static_cast<std::uint64_t>(y);
    (void)a;
}

// No cast at all.
void test_no_cast(std::uint64_t u, std::uint64_t v)
{
    bool a = u <= v;
    (void)a;
}

// The cast narrows a wider source operand, so it is not redundant.
void test_narrowing(std::uint32_t u, long long x)
{
    bool a = u <= static_cast<std::uint32_t>(x);
    (void)a;
}

// Shift operators take their result type from the left operand and promote each
// operand independently, so the cast of the shift count is not redundant.
std::uint64_t test_shift(std::uint64_t u, int x)
{
    return u << static_cast<std::uint64_t>(x);
}
