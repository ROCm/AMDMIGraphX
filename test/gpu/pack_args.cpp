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
#include <test.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <cstdint>
#include <cstring>
#include <map>

template <class T>
static std::size_t packed_sizes()
{
    return sizeof(T);
}

template <class T, class U, class... Ts>
static std::size_t packed_sizes()
{
    return sizeof(T) + packed_sizes<U, Ts...>();
}

template <class... Ts>
static std::size_t sizes()
{
    return migraphx::gpu::pack_args(std::vector<migraphx::gpu::kernel_argument>{Ts{}...}).size();
}

template <class... Ts>
static std::size_t padding()
{
    EXPECT(sizes<Ts...>() >= packed_sizes<Ts...>());
    return sizes<Ts...>() - packed_sizes<Ts...>();
}

struct float_struct
{
    float x, y;
};

TEST_CASE(alignment_padding)
{
    EXPECT(padding<short, short>() == 0);
    EXPECT(padding<float, float_struct>() == 0);
    EXPECT(padding<short, float_struct>() == 2);
    EXPECT(padding<short, int>() == 2);
    EXPECT(padding<char, short, int, char>() == 1);
}

// Store a pointer value into a kernarg buffer at a byte offset.
static void put_pointer(std::vector<char>& buf, std::size_t off, char* p)
{
    std::memcpy(buf.data() + off, &p, sizeof(char*));
}

// unpack_kernel_config with a kernel_args layout must return the offset and value
// of only the pointer slots, skipping the inlined scalar arguments. Layout below
// (matching code_object_op packing: pointer = empty value, 8 bytes align 8):
//   [0] pointer  -> offset 0
//   [1] uint32_t -> offset 8  (4 bytes)
//   [2] pointer  -> offset 16 (12 padded up to 16)
//   [3] uint16_t -> offset 24
TEST_CASE(unpack_skips_scalars)
{
    std::map<std::size_t, migraphx::gpu::kernel_argument_value> kernel_args;
    kernel_args[0] = migraphx::gpu::kernel_argument_value{};
    kernel_args[1] = migraphx::gpu::kernel_argument_value(std::uint32_t{7});
    kernel_args[2] = migraphx::gpu::kernel_argument_value{};
    kernel_args[3] = migraphx::gpu::kernel_argument_value(std::uint16_t{3});

    auto* ptr0 = reinterpret_cast<char*>(0xdead0000);
    auto* ptr2 = reinterpret_cast<char*>(0xbeef0000);
    std::vector<char> buf(32, 0);
    put_pointer(buf, 0, ptr0);
    put_pointer(buf, 16, ptr2);

    std::size_t size = buf.size();
    auto config      = migraphx::gpu::pack_kernel_config(buf.data(), &size);
    auto pointers    = migraphx::gpu::unpack_kernel_config(config.data(), kernel_args);

    EXPECT(pointers.size() == 2);
    EXPECT(pointers[0] == std::make_pair(std::size_t{0}, ptr0));
    EXPECT(pointers[1] == std::make_pair(std::size_t{16}, ptr2));
}

// An empty kernel_args is the all-pointer launch path: every 8-byte word is a
// pointer.
TEST_CASE(unpack_all_pointers)
{
    std::map<std::size_t, migraphx::gpu::kernel_argument_value> kernel_args;
    auto* ptr0 = reinterpret_cast<char*>(0x1000);
    auto* ptr1 = reinterpret_cast<char*>(0x2000);
    auto* ptr2 = reinterpret_cast<char*>(0x3000);
    std::vector<char> buf(24, 0);
    put_pointer(buf, 0, ptr0);
    put_pointer(buf, 8, ptr1);
    put_pointer(buf, 16, ptr2);

    std::size_t size = buf.size();
    auto config      = migraphx::gpu::pack_kernel_config(buf.data(), &size);
    auto pointers    = migraphx::gpu::unpack_kernel_config(config.data(), kernel_args);

    EXPECT(pointers.size() == 3);
    EXPECT(pointers[0] == std::make_pair(std::size_t{0}, ptr0));
    EXPECT(pointers[1] == std::make_pair(std::size_t{8}, ptr1));
    EXPECT(pointers[2] == std::make_pair(std::size_t{16}, ptr2));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
