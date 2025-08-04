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
#include <migraphx/byte.hpp>
#include <type_traits>
#include <sstream>
#include <limits>
#include "test.hpp"

using migraphx::byte;

TEST_CASE(byte_basic_construction)
{
    // Test default construction
    constexpr byte b1{};
    CHECK(migraphx::to_integer<std::uint8_t>(b1) == 0);

    // Test explicit construction from values
    constexpr byte b2{static_cast<byte>(42)};
    CHECK(migraphx::to_integer<std::uint8_t>(b2) == 42);

    constexpr byte b3{static_cast<byte>(255)};
    CHECK(migraphx::to_integer<std::uint8_t>(b3) == 255);

    constexpr byte b4{static_cast<byte>(128)};
    CHECK(migraphx::to_integer<std::uint8_t>(b4) == 128);
}

TEST_CASE(byte_enum_properties)
{
    // Test that byte is an enum class with unsigned char underlying type
    static_assert(std::is_enum<byte>::value, "byte should be an enum");
    static_assert(std::is_same<std::underlying_type<byte>::type, unsigned char>::value,
                  "byte should have unsigned char as underlying type");
    static_assert(std::is_same<byte, migraphx::byte>::value, "byte should be migraphx::byte");
}

TEST_CASE(byte_left_shift_operator)
{
    constexpr byte b{static_cast<byte>(0b00000001)};
    
    // Test left shift with different unsigned integer types
    CHECK(migraphx::to_integer<std::uint8_t>(b << 0u) == 0b00000001);
    CHECK(migraphx::to_integer<std::uint8_t>(b << 1u) == 0b00000010);
    CHECK(migraphx::to_integer<std::uint8_t>(b << 2u) == 0b00000100);
    CHECK(migraphx::to_integer<std::uint8_t>(b << 3u) == 0b00001000);
    CHECK(migraphx::to_integer<std::uint8_t>(b << 7u) == 0b10000000);
    
    // Test with different unsigned types
    CHECK(migraphx::to_integer<std::uint8_t>(b << static_cast<std::uint16_t>(1)) == 0b00000010);
    CHECK(migraphx::to_integer<std::uint8_t>(b << static_cast<std::uint32_t>(2)) == 0b00000100);
    CHECK(migraphx::to_integer<std::uint8_t>(b << static_cast<std::uint64_t>(3)) == 0b00001000);
    
    // Test edge case - shift by 8 or more (implementation defined behavior)
    constexpr byte b2{static_cast<byte>(0b11111111)};
    CHECK(migraphx::to_integer<std::uint8_t>(b2 << 8u) == 0);  // Should wrap around
}

TEST_CASE(byte_right_shift_operator)
{
    constexpr byte b{static_cast<byte>(0b10000000)};
    
    // Test right shift with different unsigned integer types
    CHECK(migraphx::to_integer<std::uint8_t>(b >> 0u) == 0b10000000);
    CHECK(migraphx::to_integer<std::uint8_t>(b >> 1u) == 0b01000000);
    CHECK(migraphx::to_integer<std::uint8_t>(b >> 2u) == 0b00100000);
    CHECK(migraphx::to_integer<std::uint8_t>(b >> 3u) == 0b00010000);
    CHECK(migraphx::to_integer<std::uint8_t>(b >> 7u) == 0b00000001);
    
    // Test with different unsigned types
    CHECK(migraphx::to_integer<std::uint8_t>(b >> static_cast<std::uint16_t>(1)) == 0b01000000);
    CHECK(migraphx::to_integer<std::uint8_t>(b >> static_cast<std::uint32_t>(2)) == 0b00100000);
    CHECK(migraphx::to_integer<std::uint8_t>(b >> static_cast<std::uint64_t>(3)) == 0b00010000);
    
    // Test edge case - shift by 8 or more
    CHECK(migraphx::to_integer<std::uint8_t>(b >> 8u) == 0);
}

TEST_CASE(byte_left_shift_assignment)
{
    byte b{static_cast<byte>(0b00000001)};
    
    // Test left shift assignment with different unsigned integer types
    b <<= 1u;
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b00000010);
    
    b <<= static_cast<std::uint16_t>(2);
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b00001000);
    
    b <<= static_cast<std::uint32_t>(1);
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b00010000);
    
    b <<= static_cast<std::uint64_t>(3);
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b10000000);
}

TEST_CASE(byte_right_shift_assignment)
{
    byte b{static_cast<byte>(0b10000000)};
    
    // Test right shift assignment with different unsigned integer types
    b >>= 1u;
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b01000000);
    
    b >>= static_cast<std::uint16_t>(2);
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b00010000);
    
    b >>= static_cast<std::uint32_t>(1);
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b00001000);
    
    b >>= static_cast<std::uint64_t>(3);
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b00000001);
}

TEST_CASE(byte_bitwise_or_operator)
{
    constexpr byte b1{static_cast<byte>(0b10101010)};
    constexpr byte b2{static_cast<byte>(0b01010101)};
    constexpr byte b3{static_cast<byte>(0b11110000)};
    constexpr byte b4{static_cast<byte>(0b00001111)};
    
    CHECK(migraphx::to_integer<std::uint8_t>(b1 | b2) == 0b11111111);
    CHECK(migraphx::to_integer<std::uint8_t>(b3 | b4) == 0b11111111);
    CHECK(migraphx::to_integer<std::uint8_t>(b1 | b3) == 0b11111010);
    CHECK(migraphx::to_integer<std::uint8_t>(b2 | b4) == 0b01011111);
    
    // Test with zero
    constexpr byte zero{static_cast<byte>(0)};
    CHECK(migraphx::to_integer<std::uint8_t>(b1 | zero) == 0b10101010);
    CHECK(migraphx::to_integer<std::uint8_t>(zero | b1) == 0b10101010);
}

TEST_CASE(byte_bitwise_or_assignment)
{
    byte b{static_cast<byte>(0b10101010)};
    constexpr byte mask{static_cast<byte>(0b01010101)};
    
    b |= mask;
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b11111111);
    
    byte b2{static_cast<byte>(0b11110000)};
    b2 |= byte{0b00001111};
    CHECK(migraphx::to_integer<std::uint8_t>(b2) == 0b11111111);
}

TEST_CASE(byte_bitwise_and_operator)
{
    constexpr byte b1{static_cast<byte>(0b11111111)};
    constexpr byte b2{static_cast<byte>(0b10101010)};
    constexpr byte b3{static_cast<byte>(0b11110000)};
    constexpr byte b4{static_cast<byte>(0b00001111)};
    
    CHECK(migraphx::to_integer<std::uint8_t>(b1 & b2) == 0b10101010);
    CHECK(migraphx::to_integer<std::uint8_t>(b2 & b3) == 0b10100000);
    CHECK(migraphx::to_integer<std::uint8_t>(b3 & b4) == 0b00000000);
    
    // Test with zero
    constexpr byte zero{static_cast<byte>(0)};
    CHECK(migraphx::to_integer<std::uint8_t>(b1 & zero) == 0);
    CHECK(migraphx::to_integer<std::uint8_t>(zero & b1) == 0);
}

TEST_CASE(byte_bitwise_and_assignment)
{
    byte b{static_cast<byte>(0b11111111)};
    constexpr byte mask{static_cast<byte>(0b10101010)};
    
    b &= mask;
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b10101010);
    
    byte b2{static_cast<byte>(0b11110000)};
    b2 &= byte{0b10101010};
    CHECK(migraphx::to_integer<std::uint8_t>(b2) == 0b10100000);
}

TEST_CASE(byte_bitwise_xor_operator)
{
    constexpr byte b1{static_cast<byte>(0b10101010)};
    constexpr byte b2{static_cast<byte>(0b01010101)};
    constexpr byte b3{static_cast<byte>(0b11111111)};
    
    CHECK(migraphx::to_integer<std::uint8_t>(b1 ^ b2) == 0b11111111);
    CHECK(migraphx::to_integer<std::uint8_t>(b1 ^ b3) == 0b01010101);
    CHECK(migraphx::to_integer<std::uint8_t>(b2 ^ b3) == 0b10101010);
    
    // Test XOR with self (should be zero)
    CHECK(migraphx::to_integer<std::uint8_t>(b1 ^ b1) == 0);
    CHECK(migraphx::to_integer<std::uint8_t>(b2 ^ b2) == 0);
    
    // Test XOR with zero
    constexpr byte zero{static_cast<byte>(0)};
    CHECK(migraphx::to_integer<std::uint8_t>(b1 ^ zero) == 0b10101010);
    CHECK(migraphx::to_integer<std::uint8_t>(zero ^ b1) == 0b10101010);
}

TEST_CASE(byte_bitwise_xor_assignment)
{
    byte b{static_cast<byte>(0b10101010)};
    constexpr byte mask{static_cast<byte>(0b01010101)};
    
    b ^= mask;
    CHECK(migraphx::to_integer<std::uint8_t>(b) == 0b11111111);
    
    byte b2{static_cast<byte>(0b11111111)};
    b2 ^= byte{0b10101010};
    CHECK(migraphx::to_integer<std::uint8_t>(b2) == 0b01010101);
    
    // Test XOR assignment with self (should be zero)
    byte b3{static_cast<byte>(0b10101010)};
    b3 ^= byte{0b10101010};
    CHECK(migraphx::to_integer<std::uint8_t>(b3) == 0);
}

TEST_CASE(byte_bitwise_not_operator)
{
    constexpr byte b1{static_cast<byte>(0b00000000)};
    constexpr byte b2{static_cast<byte>(0b11111111)};
    constexpr byte b3{static_cast<byte>(0b10101010)};
    constexpr byte b4{static_cast<byte>(0b01010101)};
    
    CHECK(migraphx::to_integer<std::uint8_t>(~b1) == 0b11111111);
    CHECK(migraphx::to_integer<std::uint8_t>(~b2) == 0b00000000);
    CHECK(migraphx::to_integer<std::uint8_t>(~b3) == 0b01010101);
    CHECK(migraphx::to_integer<std::uint8_t>(~b4) == 0b10101010);
    
    // Test double negation
    CHECK(migraphx::to_integer<std::uint8_t>(~~b1) == 0b00000000);
    CHECK(migraphx::to_integer<std::uint8_t>(~~b3) == 0b10101010);
}

TEST_CASE(byte_to_integer_function)
{
    constexpr byte b1{static_cast<byte>(42)};
    constexpr byte b2{static_cast<byte>(255)};
    constexpr byte b3{static_cast<byte>(0)};
    constexpr byte b4{static_cast<byte>(128)};
    
    // Test conversion to different unsigned integer types
    CHECK(migraphx::to_integer<std::uint8_t>(b1) == 42);
    CHECK(migraphx::to_integer<std::uint16_t>(b1) == 42);
    CHECK(migraphx::to_integer<std::uint32_t>(b1) == 42);
    CHECK(migraphx::to_integer<std::uint64_t>(b1) == 42);
    
    CHECK(migraphx::to_integer<std::uint8_t>(b2) == 255);
    CHECK(migraphx::to_integer<std::uint16_t>(b2) == 255);
    CHECK(migraphx::to_integer<std::uint32_t>(b2) == 255);
    CHECK(migraphx::to_integer<std::uint64_t>(b2) == 255);
    
    CHECK(migraphx::to_integer<std::uint8_t>(b3) == 0);
    CHECK(migraphx::to_integer<std::uint16_t>(b3) == 0);
    CHECK(migraphx::to_integer<std::uint32_t>(b3) == 0);
    CHECK(migraphx::to_integer<std::uint64_t>(b3) == 0);
    
    CHECK(migraphx::to_integer<std::uint8_t>(b4) == 128);
    CHECK(migraphx::to_integer<std::uint16_t>(b4) == 128);
    CHECK(migraphx::to_integer<std::uint32_t>(b4) == 128);
    CHECK(migraphx::to_integer<std::uint64_t>(b4) == 128);
}

TEST_CASE(byte_stream_insertion_operator)
{
    constexpr byte b1{static_cast<byte>(42)};   // ASCII '*'
    constexpr byte b2{static_cast<byte>(48)};   // ASCII '0'
    constexpr byte b3{static_cast<byte>(65)};   // ASCII 'A'
    constexpr byte b4{static_cast<byte>(0)};    // ASCII null
    
    std::ostringstream oss;
    
    // Test stream insertion - outputs as characters (uint8_t behavior)
    oss << b1;
    CHECK(oss.str() == "*");  // 42 is ASCII '*'
    
    oss.str("");
    oss << b2;
    CHECK(oss.str() == "0");  // 48 is ASCII '0'
    
    oss.str("");
    oss << b3;
    CHECK(oss.str() == "A");  // 65 is ASCII 'A'
    
    oss.str("");
    oss << b4;
    CHECK(oss.str().length() == 1 && oss.str()[0] == '\0');  // 0 is null character
    
    // Test multiple bytes in sequence
    oss.str("");
    oss << b2 << b3;
    CHECK(oss.str() == "0A");
}

TEST_CASE(byte_boundary_values)
{
    // Test boundary values
    constexpr byte min_byte{static_cast<byte>(0)};
    constexpr byte max_byte{static_cast<byte>(255)};
    constexpr byte mid_byte{static_cast<byte>(128)};
    
    CHECK(migraphx::to_integer<std::uint8_t>(min_byte) == 0);
    CHECK(migraphx::to_integer<std::uint8_t>(max_byte) == 255);
    CHECK(migraphx::to_integer<std::uint8_t>(mid_byte) == 128);
    
    // Test operations with boundary values
    CHECK(migraphx::to_integer<std::uint8_t>(min_byte | max_byte) == 255);
    CHECK(migraphx::to_integer<std::uint8_t>(min_byte & max_byte) == 0);
    CHECK(migraphx::to_integer<std::uint8_t>(min_byte ^ max_byte) == 255);
    CHECK(migraphx::to_integer<std::uint8_t>(~min_byte) == 255);
    CHECK(migraphx::to_integer<std::uint8_t>(~max_byte) == 0);
}

TEST_CASE(byte_shift_edge_cases)
{
    constexpr byte b{static_cast<byte>(0b10101010)};
    
    // Test shift by 0
    CHECK(migraphx::to_integer<std::uint8_t>(b << 0u) == 0b10101010);
    CHECK(migraphx::to_integer<std::uint8_t>(b >> 0u) == 0b10101010);
    
    // Test maximum meaningful shifts for 8-bit values
    constexpr byte single_bit{static_cast<byte>(0b00000001)};
    CHECK(migraphx::to_integer<std::uint8_t>(single_bit << 7u) == 0b10000000);
    
    constexpr byte msb_set{static_cast<byte>(0b10000000)};
    CHECK(migraphx::to_integer<std::uint8_t>(msb_set >> 7u) == 0b00000001);
    
    // Test shifts that result in zero
    CHECK(migraphx::to_integer<std::uint8_t>(single_bit << 8u) == 0);
    CHECK(migraphx::to_integer<std::uint8_t>(msb_set >> 8u) == 0);
}

TEST_CASE(byte_constexpr_evaluation)
{
    // Test that operations can be evaluated at compile time
    constexpr byte b1{static_cast<byte>(42)};
    constexpr byte b2{static_cast<byte>(128)};
    
    constexpr auto or_result = b1 | b2;
    constexpr auto and_result = b1 & b2;
    constexpr auto xor_result = b1 ^ b2;
    constexpr auto not_result = ~b1;
    constexpr auto shift_left_result = b1 << 2u;
    constexpr auto shift_right_result = b2 >> 1u;
    
    static_assert(migraphx::to_integer<std::uint8_t>(or_result) == (42 | 128), 
                  "Constexpr OR evaluation failed");
    static_assert(migraphx::to_integer<std::uint8_t>(and_result) == (42 & 128), 
                  "Constexpr AND evaluation failed");
    static_assert(migraphx::to_integer<std::uint8_t>(xor_result) == (42 ^ 128), 
                  "Constexpr XOR evaluation failed");
    static_assert(migraphx::to_integer<std::uint8_t>(not_result) == (~42u & 0xFF), 
                  "Constexpr NOT evaluation failed");
    static_assert(migraphx::to_integer<std::uint8_t>(shift_left_result) == ((42 << 2) & 0xFF), 
                  "Constexpr left shift evaluation failed");
    static_assert(migraphx::to_integer<std::uint8_t>(shift_right_result) == (128 >> 1), 
                  "Constexpr right shift evaluation failed");
}

TEST_CASE(byte_various_bit_patterns)
{
    // Test with various bit patterns
    constexpr byte all_ones{static_cast<byte>(0b11111111)};
    constexpr byte all_zeros{static_cast<byte>(0b00000000)};
    constexpr byte alternating1{static_cast<byte>(0b10101010)};
    constexpr byte alternating2{static_cast<byte>(0b01010101)};
    constexpr byte lower_nibble{static_cast<byte>(0b00001111)};
    constexpr byte upper_nibble{static_cast<byte>(0b11110000)};
    
    // Test OR operations
    CHECK(migraphx::to_integer<std::uint8_t>(alternating1 | alternating2) == 0b11111111);
    CHECK(migraphx::to_integer<std::uint8_t>(lower_nibble | upper_nibble) == 0b11111111);
    
    // Test AND operations
    CHECK(migraphx::to_integer<std::uint8_t>(alternating1 & alternating2) == 0b00000000);
    CHECK(migraphx::to_integer<std::uint8_t>(lower_nibble & upper_nibble) == 0b00000000);
    CHECK(migraphx::to_integer<std::uint8_t>(all_ones & alternating1) == 0b10101010);
    
    // Test XOR operations
    CHECK(migraphx::to_integer<std::uint8_t>(alternating1 ^ alternating2) == 0b11111111);
    CHECK(migraphx::to_integer<std::uint8_t>(lower_nibble ^ upper_nibble) == 0b11111111);
    CHECK(migraphx::to_integer<std::uint8_t>(all_ones ^ alternating1) == 0b01010101);
    
    // Test NOT operations
    CHECK(migraphx::to_integer<std::uint8_t>(~alternating1) == 0b01010101);
    CHECK(migraphx::to_integer<std::uint8_t>(~alternating2) == 0b10101010);
    CHECK(migraphx::to_integer<std::uint8_t>(~lower_nibble) == 0b11110000);
    CHECK(migraphx::to_integer<std::uint8_t>(~upper_nibble) == 0b00001111);
}

// Note: Type safety tests cannot be implemented as proper unit tests because
// they would result in compilation errors. The following commented tests
// demonstrate what should NOT compile:
/*
TEST_CASE(byte_type_safety)
{
    constexpr byte b1{static_cast<byte>(42)};
    constexpr byte b2{static_cast<byte>(128)};
    
    // These operations should NOT compile (arithmetic operations not allowed):
    // auto sum = b1 + b2;        // Should not compile
    // auto diff = b2 - b1;       // Should not compile
    // auto prod = b1 * b2;       // Should not compile
    // auto quot = b2 / b1;       // Should not compile
    // auto mod = b2 % b1;        // Should not compile
    // ++b1;                      // Should not compile
    // b1++;                      // Should not compile
    // --b1;                      // Should not compile
    // b1--;                      // Should not compile
    // b1 += b2;                  // Should not compile
    // b1 -= b2;                  // Should not compile
    // b1 *= b2;                  // Should not compile
    // b1 /= b2;                  // Should not compile
    // b1 %= b2;                  // Should not compile
}
*/