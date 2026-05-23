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
#include <migraphx/md5.hpp>
#include "test.hpp"
#include <string>
#include <string_view>

// RFC 1321 test vectors (appendix A.5).
TEST_CASE(md5_rfc_empty) { EXPECT(migraphx::md5("") == "d41d8cd98f00b204e9800998ecf8427e"); }

TEST_CASE(md5_rfc_a) { EXPECT(migraphx::md5("a") == "0cc175b9c0f1b6a831c399e269772661"); }

TEST_CASE(md5_rfc_abc) { EXPECT(migraphx::md5("abc") == "900150983cd24fb0d6963f7d28e17f72"); }

TEST_CASE(md5_rfc_message_digest)
{
    EXPECT(migraphx::md5("message digest") == "f96b697d7cb7938d525a2f31aaf161d0");
}

TEST_CASE(md5_rfc_lowercase_alphabet)
{
    EXPECT(migraphx::md5("abcdefghijklmnopqrstuvwxyz") == "c3fcd3d76192e4007dfb496cca67e13b");
}

TEST_CASE(md5_rfc_alphanumeric)
{
    EXPECT(migraphx::md5("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") ==
           "d174ab98d277d9f5a5611c2c9f419d9f");
}

TEST_CASE(md5_rfc_digits)
{
    EXPECT(migraphx::md5("1234567890123456789012345678901234567890"
                         "1234567890123456789012345678901234567890") ==
           "57edf4a22be3c955ac49da2e2107b67a");
}

// Well-known extra test strings.
TEST_CASE(md5_quick_brown_fox)
{
    EXPECT(migraphx::md5("The quick brown fox jumps over the lazy dog") ==
           "9e107d9d372bb6826bd81d3542a419d6");
}

TEST_CASE(md5_quick_brown_fox_period)
{
    EXPECT(migraphx::md5("The quick brown fox jumps over the lazy dog.") ==
           "e4d909c290d0fb1ca068ffaddf22cbd0");
}

// Block boundary tests. MD5 processes in 64-byte blocks; the bit length field
// must fit in the trailing 8 bytes of the last padding block, so input length
// modulo 64 being 55/56/63/64 exercises all padding branches.
TEST_CASE(md5_length_55)
{
    EXPECT(migraphx::md5(std::string(55, 'a')) == "ef1772b6dff9a122358552954ad0df65");
}

TEST_CASE(md5_length_56)
{
    EXPECT(migraphx::md5(std::string(56, 'a')) == "3b0c8ac703f828b04c6c197006d17218");
}

TEST_CASE(md5_length_63)
{
    EXPECT(migraphx::md5(std::string(63, 'a')) == "b06521f39153d618550606be297466d5");
}

TEST_CASE(md5_length_64)
{
    EXPECT(migraphx::md5(std::string(64, 'a')) == "014842d480b571495a4a0363793f7367");
}

TEST_CASE(md5_length_65)
{
    EXPECT(migraphx::md5(std::string(65, 'a')) == "c743a45e0d2e6a95cb859adae0248435");
}

TEST_CASE(md5_length_119)
{
    EXPECT(migraphx::md5(std::string(119, 'a')) == "8a7bd0732ed6a28ce75f6dabc90e1613");
}

TEST_CASE(md5_length_120)
{
    EXPECT(migraphx::md5(std::string(120, 'a')) == "5f61c0ccad4cac44c75ff505e1f1e537");
}

// Embedded NUL bytes must be hashed, not treated as terminators.
TEST_CASE(md5_embedded_null)
{
    const std::string s{"a\0b", 3};
    EXPECT(migraphx::md5(s) == "70350f6027bce3713f6b76473084309b");
}

TEST_CASE(md5_trailing_null)
{
    EXPECT(migraphx::md5(std::string(1, '\0')) == "93b885adfe0da089cdf634904fd59f71");
}

// High-byte input — make sure bytes aren't sign-extended.
TEST_CASE(md5_high_bytes)
{
    const std::string s(16, '\xff');
    EXPECT(migraphx::md5(s) == "8d79cbc9a4ecdde112fc91ba625b13c2");
}

// Output shape: 32 lowercase hex characters.
TEST_CASE(md5_output_format)
{
    const std::string hash = migraphx::md5("anything");
    EXPECT(hash.size() == 32);
    for(char c : hash)
    {
        const bool is_hex = (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f');
        EXPECT(is_hex);
    }
}

// Determinism: same input yields same digest every call.
TEST_CASE(md5_deterministic)
{
    const std::string input = "repeatable input";
    EXPECT(migraphx::md5(input) == migraphx::md5(input));
}

// Small perturbations of input yield different digests (avalanche sanity check).
TEST_CASE(md5_distinct_outputs)
{
    EXPECT(migraphx::md5("abc") != migraphx::md5("abd"));
    EXPECT(migraphx::md5("abc") != migraphx::md5("ABC"));
    EXPECT(migraphx::md5("") != migraphx::md5(" "));
    EXPECT(migraphx::md5("a") != migraphx::md5("aa"));
}

// Accepts string_view constructed from char buffer.
TEST_CASE(md5_string_view_from_buffer)
{
    const char buf[]            = "abc";
    const std::string_view view = {buf, 3};
    EXPECT(migraphx::md5(view) == "900150983cd24fb0d6963f7d28e17f72");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
