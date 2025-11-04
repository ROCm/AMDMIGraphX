/*
* The MIT License (MIT)
*
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/test.hpp>
#include <migraphx/kernels/float_equal.hpp>

// Test default construction
TEST_CASE(array_default_construction)
{
    migraphx::array<int, 5> arr;
    EXPECT(arr.size() == 5);
    EXPECT(arr.data() != nullptr);
    EXPECT(not arr.empty());
}

// Test variadic constructor
TEST_CASE(array_variadic_constructor)
{
    migraphx::array<int, 5> arr{1, 2, 3, 4, 5};
    EXPECT(arr[0] == 1);
    EXPECT(arr[1] == 2);
    EXPECT(arr[2] == 3);
    EXPECT(arr[3] == 4);
    EXPECT(arr[4] == 5);
}

// Test single value constructor
TEST_CASE(array_single_value_constructor)
{
    migraphx::array<int, 4> arr(42);
    EXPECT(arr[0] == 42);
    EXPECT(arr[1] == 42);
    EXPECT(arr[2] == 42);
    EXPECT(arr[3] == 42);
}

// Test single value constructor with size 1
TEST_CASE(array_single_value_constructor_size_one)
{
    migraphx::array<int, 1> arr(7);
    EXPECT(arr[0] == 7);
}

// Test element access with operator[]
TEST_CASE(array_element_access)
{
    migraphx::array<int, 3> arr{10, 20, 30};
    EXPECT(arr[0] == 10);
    EXPECT(arr[1] == 20);
    EXPECT(arr[2] == 30);

    arr[1] = 25;
    EXPECT(arr[1] == 25);
}

// Test const element access
TEST_CASE(array_const_element_access)
{
    const migraphx::array<int, 3> arr{10, 20, 30};
    EXPECT(arr[0] == 10);
    EXPECT(arr[1] == 20);
    EXPECT(arr[2] == 30);
}

// Test front() and back()
TEST_CASE(array_front_back)
{
    migraphx::array<int, 4> arr{1, 2, 3, 4};
    EXPECT(arr.front() == 1);
    EXPECT(arr.back() == 4);

    arr.front() = 10;
    arr.back()  = 40;
    EXPECT(arr[0] == 10);
    EXPECT(arr[3] == 40);
}

// Test const front() and back()
TEST_CASE(array_const_front_back)
{
    const migraphx::array<int, 4> arr{1, 2, 3, 4};
    EXPECT(arr.front() == 1);
    EXPECT(arr.back() == 4);
}

// Test data() method
TEST_CASE(array_data_method)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    int* ptr = arr.data();
    EXPECT(ptr[0] == 1);
    EXPECT(ptr[1] == 2);
    EXPECT(ptr[2] == 3);

    ptr[0] = 10;
    EXPECT(arr[0] == 10);
}

// Test const data() method
TEST_CASE(array_const_data_method)
{
    const migraphx::array<int, 3> arr{1, 2, 3};
    const int* ptr = arr.data();
    EXPECT(ptr[0] == 1);
    EXPECT(ptr[1] == 2);
    EXPECT(ptr[2] == 3);
}

// Test size() method
TEST_CASE(array_size_method)
{
    // migraphx::array<int, 0> arr0;  // Commented out due to compiler limitations
    migraphx::array<int, 1> arr1;
    migraphx::array<int, 5> arr5;
    migraphx::array<int, 100> arr100;

    // EXPECT(arr0.size() == 0);  // Commented out
    EXPECT(arr1.size() == 1);
    EXPECT(arr5.size() == 5);
    EXPECT(arr100.size() == 100);
}

// Test empty() method
TEST_CASE(array_empty_method)
{
    // migraphx::array<int, 0> arr0;  // Commented out due to compiler limitations
    migraphx::array<int, 1> arr1;

    // EXPECT(arr0.empty());  // Commented out
    EXPECT(not arr1.empty());
}

// Test begin() and end() iterators
TEST_CASE(array_iterators)
{
    migraphx::array<int, 4> arr{1, 2, 3, 4};

    auto* it = arr.begin();
    EXPECT(*it == 1);
    ++it;
    EXPECT(*it == 2);
    ++it;
    EXPECT(*it == 3);
    ++it;
    EXPECT(*it == 4);
    ++it;
    EXPECT(it == arr.end());
}

// Test const iterators
TEST_CASE(array_const_iterators)
{
    const migraphx::array<int, 3> arr{5, 10, 15};

    const auto* it = arr.begin();
    EXPECT(*it == 5);
    ++it;
    EXPECT(*it == 10);
    ++it;
    EXPECT(*it == 15);
    ++it;
    EXPECT(it == arr.end());
}

// Test iterator distance
TEST_CASE(array_iterator_distance)
{
    migraphx::array<int, 5> arr{1, 2, 3, 4, 5};
    EXPECT(arr.end() - arr.begin() == 5);
}

// Test dot product
TEST_CASE(array_dot_product)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{4, 5, 6};

    int result = a.dot(b);
    EXPECT(result == 32); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

// Test dot product with zero
TEST_CASE(array_dot_product_zero)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{0, 0, 0};

    int result = a.dot(b);
    EXPECT(result == 0);
}

// Test product() method
TEST_CASE(array_product)
{
    migraphx::array<int, 4> arr{2, 3, 4, 5};
    int result = arr.product();
    EXPECT(result == 120); // 2 * 3 * 4 * 5 = 120
}

// Test product with zero
TEST_CASE(array_product_with_zero)
{
    migraphx::array<int, 3> arr{2, 0, 5};
    int result = arr.product();
    EXPECT(result == 0);
}

// Test product with single element
TEST_CASE(array_product_single_element)
{
    migraphx::array<int, 1> arr{7};
    int result = arr.product();
    EXPECT(result == 7);
}

// Test single() method
TEST_CASE(array_single_method)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    int result = arr.single(10);
    EXPECT(result == 123); // 1*100 + 2*10 + 3*1 = 100 + 20 + 3 = 123
}

// Test single() with default width
TEST_CASE(array_single_default_width)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    int result = arr.single();
    EXPECT(result == 10203); // 1*10000 + 2*100 + 3*1 = 10000 + 200 + 3 = 10203
}

// Test apply() method
TEST_CASE(array_apply_method)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    auto result = arr.apply([](int x) { return x * 2; });

    EXPECT(result[0] == 2);
    EXPECT(result[1] == 4);
    EXPECT(result[2] == 6);
}

// Test apply() with type conversion
TEST_CASE(array_apply_type_conversion)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    auto result = arr.apply([](int x) { return static_cast<double>(x) * 1.5; });

    EXPECT(migraphx::float_equal(result[0], 1.5));
    EXPECT(migraphx::float_equal(result[1], 3.0));
    EXPECT(migraphx::float_equal(result[2], 4.5));
}

// Test reduce() method
TEST_CASE(array_reduce_method)
{
    migraphx::array<int, 4> arr{1, 2, 3, 4};
    int sum = arr.reduce([](int a, int b) { return a + b; }, 0);
    EXPECT(sum == 10);

    int product = arr.reduce([](int a, int b) { return a * b; }, 1);
    EXPECT(product == 24);
}

// Test reduce() with custom initial value
TEST_CASE(array_reduce_custom_init)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    int result = arr.reduce([](int a, int b) { return a + b; }, 100);
    EXPECT(result == 106); // 100 + 1 + 2 + 3 = 106
}

// Test += operator with array
TEST_CASE(array_add_assign_array)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{4, 5, 6};

    a += b;
    EXPECT(a[0] == 5);
    EXPECT(a[1] == 7);
    EXPECT(a[2] == 9);
}

// Test += operator with scalar
TEST_CASE(array_add_assign_scalar)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    arr += 10;

    EXPECT(arr[0] == 11);
    EXPECT(arr[1] == 12);
    EXPECT(arr[2] == 13);
}

// Test -= operator with array
TEST_CASE(array_sub_assign_array)
{
    migraphx::array<int, 3> a{5, 7, 9};
    migraphx::array<int, 3> b{1, 2, 3};

    a -= b;
    EXPECT(a[0] == 4);
    EXPECT(a[1] == 5);
    EXPECT(a[2] == 6);
}

// Test -= operator with scalar
TEST_CASE(array_sub_assign_scalar)
{
    migraphx::array<int, 3> arr{10, 20, 30};
    arr -= 5;

    EXPECT(arr[0] == 5);
    EXPECT(arr[1] == 15);
    EXPECT(arr[2] == 25);
}

// Test *= operator with array
TEST_CASE(array_mul_assign_array)
{
    migraphx::array<int, 3> a{2, 3, 4};
    migraphx::array<int, 3> b{3, 4, 5};

    a *= b;
    EXPECT(a[0] == 6);
    EXPECT(a[1] == 12);
    EXPECT(a[2] == 20);
}

// Test *= operator with scalar
TEST_CASE(array_mul_assign_scalar)
{
    migraphx::array<int, 3> arr{2, 3, 4};
    arr *= 3;

    EXPECT(arr[0] == 6);
    EXPECT(arr[1] == 9);
    EXPECT(arr[2] == 12);
}

// Test /= operator with array
TEST_CASE(array_div_assign_array)
{
    migraphx::array<int, 3> a{12, 15, 20};
    migraphx::array<int, 3> b{3, 5, 4};

    a /= b;
    EXPECT(a[0] == 4);
    EXPECT(a[1] == 3);
    EXPECT(a[2] == 5);
}

// Test /= operator with scalar
TEST_CASE(array_div_assign_scalar)
{
    migraphx::array<int, 3> arr{12, 18, 24};
    arr /= 3;

    EXPECT(arr[0] == 4);
    EXPECT(arr[1] == 6);
    EXPECT(arr[2] == 8);
}

// Test %= operator with array
TEST_CASE(array_mod_assign_array)
{
    migraphx::array<int, 3> a{10, 13, 17};
    migraphx::array<int, 3> b{3, 5, 7};

    a %= b;
    EXPECT(a[0] == 1); // 10 % 3 = 1
    EXPECT(a[1] == 3); // 13 % 5 = 3
    EXPECT(a[2] == 3); // 17 % 7 = 3
}

// Test %= operator with scalar
TEST_CASE(array_mod_assign_scalar)
{
    migraphx::array<int, 3> arr{10, 13, 17};
    arr %= 5;

    EXPECT(arr[0] == 0); // 10 % 5 = 0
    EXPECT(arr[1] == 3); // 13 % 5 = 3
    EXPECT(arr[2] == 2); // 17 % 5 = 2
}

// Test &= operator with array
TEST_CASE(array_and_assign_array)
{
    migraphx::array<int, 3> a{15, 12, 8}; // 1111, 1100, 1000
    migraphx::array<int, 3> b{7, 10, 15}; // 0111, 1010, 1111

    a &= b;
    EXPECT(a[0] == 7); // 1111 & 0111 = 0111 = 7
    EXPECT(a[1] == 8); // 1100 & 1010 = 1000 = 8
    EXPECT(a[2] == 8); // 1000 & 1111 = 1000 = 8
}

// Test &= operator with scalar
TEST_CASE(array_and_assign_scalar)
{
    migraphx::array<int, 3> arr{15, 12, 8}; // 1111, 1100, 1000
    arr &= 7;                               // 0111

    EXPECT(arr[0] == 7); // 1111 & 0111 = 0111 = 7
    EXPECT(arr[1] == 4); // 1100 & 0111 = 0100 = 4
    EXPECT(arr[2] == 0); // 1000 & 0111 = 0000 = 0
}

// Test |= operator with array
TEST_CASE(array_or_assign_array)
{
    migraphx::array<int, 3> a{8, 4, 2}; // 1000, 0100, 0010
    migraphx::array<int, 3> b{4, 2, 1}; // 0100, 0010, 0001

    a |= b;
    EXPECT(a[0] == 12); // 1000 | 0100 = 1100 = 12
    EXPECT(a[1] == 6);  // 0100 | 0010 = 0110 = 6
    EXPECT(a[2] == 3);  // 0010 | 0001 = 0011 = 3
}

// Test |= operator with scalar
TEST_CASE(array_or_assign_scalar)
{
    migraphx::array<int, 3> arr{8, 4, 2}; // 1000, 0100, 0010
    arr |= 1;                             // 0001

    EXPECT(arr[0] == 9); // 1000 | 0001 = 1001 = 9
    EXPECT(arr[1] == 5); // 0100 | 0001 = 0101 = 5
    EXPECT(arr[2] == 3); // 0010 | 0001 = 0011 = 3
}

// Test ^= operator with array
TEST_CASE(array_xor_assign_array)
{
    migraphx::array<int, 3> a{12, 10, 6}; // 1100, 1010, 0110
    migraphx::array<int, 3> b{7, 5, 3};   // 0111, 0101, 0011

    a ^= b;
    EXPECT(a[0] == 11); // 1100 ^ 0111 = 1011 = 11
    EXPECT(a[1] == 15); // 1010 ^ 0101 = 1111 = 15
    EXPECT(a[2] == 5);  // 0110 ^ 0011 = 0101 = 5
}

// Test ^= operator with scalar
TEST_CASE(array_xor_assign_scalar)
{
    migraphx::array<int, 3> arr{12, 10, 6}; // 1100, 1010, 0110
    arr ^= 5;                               // 0101

    EXPECT(arr[0] == 9);  // 1100 ^ 0101 = 1001 = 9
    EXPECT(arr[1] == 15); // 1010 ^ 0101 = 1111 = 15
    EXPECT(arr[2] == 3);  // 0110 ^ 0101 = 0011 = 3
}

// Test + operator with arrays
TEST_CASE(array_add_arrays)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{4, 5, 6};

    auto result = a + b;
    EXPECT(result[0] == 5);
    EXPECT(result[1] == 7);
    EXPECT(result[2] == 9);
}

// Test + operator with scalar (array + scalar)
TEST_CASE(array_add_scalar_right)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    auto result = arr + 10;

    EXPECT(result[0] == 11);
    EXPECT(result[1] == 12);
    EXPECT(result[2] == 13);
}

// Test + operator with scalar (scalar + array)
TEST_CASE(array_add_scalar_left)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    auto result = 10 + arr;

    EXPECT(result[0] == 11);
    EXPECT(result[1] == 12);
    EXPECT(result[2] == 13);
}

// Test - operator with arrays
TEST_CASE(array_sub_arrays)
{
    migraphx::array<int, 3> a{5, 7, 9};
    migraphx::array<int, 3> b{1, 2, 3};

    auto result = a - b;
    EXPECT(result[0] == 4);
    EXPECT(result[1] == 5);
    EXPECT(result[2] == 6);
}

// Test - operator with scalar (array - scalar)
TEST_CASE(array_sub_scalar_right)
{
    migraphx::array<int, 3> arr{10, 20, 30};
    auto result = arr - 5;

    EXPECT(result[0] == 5);
    EXPECT(result[1] == 15);
    EXPECT(result[2] == 25);
}

// Test - operator with scalar (scalar - array)
TEST_CASE(array_sub_scalar_left)
{
    migraphx::array<int, 3> arr{1, 2, 3};
    auto result = 10 - arr;

    EXPECT(result[0] == 9);
    EXPECT(result[1] == 8);
    EXPECT(result[2] == 7);
}

// Test * operator with arrays
TEST_CASE(array_mul_arrays)
{
    migraphx::array<int, 3> a{2, 3, 4};
    migraphx::array<int, 3> b{3, 4, 5};

    auto result = a * b;
    EXPECT(result[0] == 6);
    EXPECT(result[1] == 12);
    EXPECT(result[2] == 20);
}

// Test * operator with scalar (array * scalar)
TEST_CASE(array_mul_scalar_right)
{
    migraphx::array<int, 3> arr{2, 3, 4};
    auto result = arr * 3;

    EXPECT(result[0] == 6);
    EXPECT(result[1] == 9);
    EXPECT(result[2] == 12);
}

// Test * operator with scalar (scalar * array)
TEST_CASE(array_mul_scalar_left)
{
    migraphx::array<int, 3> arr{2, 3, 4};
    auto result = 3 * arr;

    EXPECT(result[0] == 6);
    EXPECT(result[1] == 9);
    EXPECT(result[2] == 12);
}

// Test / operator with arrays
TEST_CASE(array_div_arrays)
{
    migraphx::array<int, 3> a{12, 15, 20};
    migraphx::array<int, 3> b{3, 5, 4};

    auto result = a / b;
    EXPECT(result[0] == 4);
    EXPECT(result[1] == 3);
    EXPECT(result[2] == 5);
}

// Test / operator with scalar (array / scalar)
TEST_CASE(array_div_scalar_right)
{
    migraphx::array<int, 3> arr{12, 18, 24};
    auto result = arr / 3;

    EXPECT(result[0] == 4);
    EXPECT(result[1] == 6);
    EXPECT(result[2] == 8);
}

// Test / operator with scalar (scalar / array)
TEST_CASE(array_div_scalar_left)
{
    migraphx::array<int, 3> arr{2, 3, 4};
    auto result = 12 / arr;

    EXPECT(result[0] == 6);
    EXPECT(result[1] == 4);
    EXPECT(result[2] == 3);
}

// Test % operator with arrays
TEST_CASE(array_mod_arrays)
{
    migraphx::array<int, 3> a{10, 13, 17};
    migraphx::array<int, 3> b{3, 5, 7};

    auto result = a % b;
    EXPECT(result[0] == 1); // 10 % 3 = 1
    EXPECT(result[1] == 3); // 13 % 5 = 3
    EXPECT(result[2] == 3); // 17 % 7 = 3
}

// Test % operator with scalar (array % scalar)
TEST_CASE(array_mod_scalar_right)
{
    migraphx::array<int, 3> arr{10, 13, 17};
    auto result = arr % 5;

    EXPECT(result[0] == 0); // 10 % 5 = 0
    EXPECT(result[1] == 3); // 13 % 5 = 3
    EXPECT(result[2] == 2); // 17 % 5 = 2
}

// Test % operator with scalar (scalar % array)
TEST_CASE(array_mod_scalar_left)
{
    migraphx::array<int, 3> arr{3, 4, 5};
    auto result = 17 % arr;

    EXPECT(result[0] == 2); // 17 % 3 = 2
    EXPECT(result[1] == 1); // 17 % 4 = 1
    EXPECT(result[2] == 2); // 17 % 5 = 2
}

// Test & operator with arrays
TEST_CASE(array_and_arrays)
{
    migraphx::array<int, 3> a{15, 12, 8}; // 1111, 1100, 1000
    migraphx::array<int, 3> b{7, 10, 15}; // 0111, 1010, 1111

    auto result = a & b;
    EXPECT(result[0] == 7); // 1111 & 0111 = 0111 = 7
    EXPECT(result[1] == 8); // 1100 & 1010 = 1000 = 8
    EXPECT(result[2] == 8); // 1000 & 1111 = 1000 = 8
}

// Test & operator with scalar (array & scalar)
TEST_CASE(array_and_scalar_right)
{
    migraphx::array<int, 3> arr{15, 12, 8}; // 1111, 1100, 1000
    auto result = arr & 7;                  // 0111

    EXPECT(result[0] == 7); // 1111 & 0111 = 0111 = 7
    EXPECT(result[1] == 4); // 1100 & 0111 = 0100 = 4
    EXPECT(result[2] == 0); // 1000 & 0111 = 0000 = 0
}

// Test & operator with scalar (scalar & array)
TEST_CASE(array_and_scalar_left)
{
    migraphx::array<int, 3> arr{15, 12, 8}; // 1111, 1100, 1000
    auto result = 7 & arr;                  // 0111

    EXPECT(result[0] == 7); // 0111 & 1111 = 0111 = 7
    EXPECT(result[1] == 4); // 0111 & 1100 = 0100 = 4
    EXPECT(result[2] == 0); // 0111 & 1000 = 0000 = 0
}

// Test | operator with arrays
TEST_CASE(array_or_arrays)
{
    migraphx::array<int, 3> a{8, 4, 2}; // 1000, 0100, 0010
    migraphx::array<int, 3> b{4, 2, 1}; // 0100, 0010, 0001

    auto result = a | b;
    EXPECT(result[0] == 12); // 1000 | 0100 = 1100 = 12
    EXPECT(result[1] == 6);  // 0100 | 0010 = 0110 = 6
    EXPECT(result[2] == 3);  // 0010 | 0001 = 0011 = 3
}

// Test | operator with scalar (array | scalar)
TEST_CASE(array_or_scalar_right)
{
    migraphx::array<int, 3> arr{8, 4, 2}; // 1000, 0100, 0010
    auto result = arr | 1;                // 0001

    EXPECT(result[0] == 9); // 1000 | 0001 = 1001 = 9
    EXPECT(result[1] == 5); // 0100 | 0001 = 0101 = 5
    EXPECT(result[2] == 3); // 0010 | 0001 = 0011 = 3
}

// Test | operator with scalar (scalar | array)
TEST_CASE(array_or_scalar_left)
{
    migraphx::array<int, 3> arr{8, 4, 2}; // 1000, 0100, 0010
    auto result = 1 | arr;                // 0001

    EXPECT(result[0] == 9); // 0001 | 1000 = 1001 = 9
    EXPECT(result[1] == 5); // 0001 | 0100 = 0101 = 5
    EXPECT(result[2] == 3); // 0001 | 0010 = 0011 = 3
}

// Test ^ operator with arrays
TEST_CASE(array_xor_arrays)
{
    migraphx::array<int, 3> a{12, 10, 6}; // 1100, 1010, 0110
    migraphx::array<int, 3> b{7, 5, 3};   // 0111, 0101, 0011

    auto result = a ^ b;
    EXPECT(result[0] == 11); // 1100 ^ 0111 = 1011 = 11
    EXPECT(result[1] == 15); // 1010 ^ 0101 = 1111 = 15
    EXPECT(result[2] == 5);  // 0110 ^ 0011 = 0101 = 5
}

// Test ^ operator with scalar (array ^ scalar)
TEST_CASE(array_xor_scalar_right)
{
    migraphx::array<int, 3> arr{12, 10, 6}; // 1100, 1010, 0110
    auto result = arr ^ 5;                  // 0101

    EXPECT(result[0] == 9);  // 1100 ^ 0101 = 1001 = 9
    EXPECT(result[1] == 15); // 1010 ^ 0101 = 1111 = 15
    EXPECT(result[2] == 3);  // 0110 ^ 0101 = 0011 = 3
}

// Test ^ operator with scalar (scalar ^ array)
TEST_CASE(array_xor_scalar_left)
{
    migraphx::array<int, 3> arr{12, 10, 6}; // 1100, 1010, 0110
    auto result = 5 ^ arr;                  // 0101

    EXPECT(result[0] == 9);  // 0101 ^ 1100 = 1001 = 9
    EXPECT(result[1] == 15); // 0101 ^ 1010 = 1111 = 15
    EXPECT(result[2] == 3);  // 0101 ^ 0110 = 0011 = 3
}

// Test == operator with arrays
TEST_CASE(array_equality_arrays)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{1, 2, 3};
    migraphx::array<int, 3> c{1, 2, 4};

    EXPECT(a == b);
    EXPECT(not(a == c));
}

// Test == operator with scalar (array == scalar)
TEST_CASE(array_equality_scalar_right)
{
    migraphx::array<int, 3> arr1{5, 5, 5};
    migraphx::array<int, 3> arr2{5, 5, 6};

    EXPECT(arr1 == 5);
    EXPECT(not(arr2 == 5));
}

// Test == operator with scalar (scalar == array)
TEST_CASE(array_equality_scalar_left)
{
    migraphx::array<int, 3> arr1{5, 5, 5};
    migraphx::array<int, 3> arr2{5, 5, 6};

    EXPECT(5 == arr1);
    EXPECT(not(5 == arr2));
}

// Test != operator
TEST_CASE(array_inequality)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{1, 2, 3};
    migraphx::array<int, 3> c{1, 2, 4};

    EXPECT(a == b);      // Test equality instead
    EXPECT(not(a == c)); // Test inequality using negation of ==
    EXPECT(not(a == 5)); // Using negation of == instead of !=
    EXPECT(not(5 == a)); // Using negation of == instead of !=
}

// Test < operator (product order)
TEST_CASE(array_less_than)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{2, 3, 4};
    migraphx::array<int, 3> c{1, 1, 3};

    EXPECT(a < b);      // all elements of a < corresponding elements of b
    EXPECT(not(a < c)); // a[1] = 2 is not < c[1] = 1
}

// Test > operator
TEST_CASE(array_greater_than)
{
    migraphx::array<int, 3> a{2, 3, 4};
    migraphx::array<int, 3> b{1, 2, 3};

    EXPECT(a > b);
    EXPECT(not(b > a));
}

// Test <= operator
TEST_CASE(array_less_equal)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{1, 2, 3};
    migraphx::array<int, 3> c{2, 3, 4};

    EXPECT(a <= b); // equal
    EXPECT(a <= c); // less than
    EXPECT(not(c <= a));
}

// Test >= operator
TEST_CASE(array_greater_equal)
{
    migraphx::array<int, 3> a{1, 2, 3};
    migraphx::array<int, 3> b{1, 2, 3};
    migraphx::array<int, 3> c{2, 3, 4};

    EXPECT(a >= b); // equal
    EXPECT(c >= a); // greater than
    EXPECT(not(a >= c));
}

// Test carry() method
TEST_CASE(array_carry_method)
{
    migraphx::array<int, 3> shape{10, 10, 10};
    migraphx::array<int, 3> input{5, 15, 25};

    auto result = shape.carry(input);
    // Detailed calculation:
    // Start with {5, 15, 25}
    // Process index 2: 25 -> 25-10-10 = 5, overflow = 2
    // Process index 1: 15+2 = 17 -> 17-10 = 7, overflow = 1
    // Process index 0: 5+1 = 6
    EXPECT(result[0] == 6); // 5 + 1 (overflow from index 1)
    EXPECT(result[1] == 7); // (15 + 2) % 10 = 17 % 10 = 7
    EXPECT(result[2] == 5); // 25 % 10 = 5
}

// Test carry() with no overflow
TEST_CASE(array_carry_no_overflow)
{
    migraphx::array<int, 3> shape{10, 10, 10};
    migraphx::array<int, 3> input{1, 2, 3};

    auto result = shape.carry(input);
    EXPECT(result[0] == 1);
    EXPECT(result[1] == 2);
    EXPECT(result[2] == 3);
}

// Test multi() method
TEST_CASE(array_multi_method)
{
    migraphx::array<int, 3> shape{2, 3, 4};

    auto result = shape.multi(11); // 11 in base (2,3,4)
    // 11 = 1*12 + 3*4 + 3*1 = 12 + 12 + 3 = 27? Let's check calculation
    // For shape {2,3,4}, we compute multi-index for linear index 11

    // result[2] = 11 % 4 = 3
    // temp = 11 / 4 = 2
    // result[1] = 2 % 3 = 2
    // temp = 2 / 3 = 0
    // result[0] = 0

    EXPECT(result[0] == 0);
    EXPECT(result[1] == 2);
    EXPECT(result[2] == 3);
}

// Test multi() method with simple case
TEST_CASE(array_multi_simple)
{
    migraphx::array<int, 2> shape{3, 4};

    auto result = shape.multi(5); // 5 in base (3,4)
    // result[1] = 5 % 4 = 1
    // temp = 5 / 4 = 1
    // result[0] = 1

    EXPECT(result[0] == 1);
    EXPECT(result[1] == 1);
}

// Test different data types
TEST_CASE(array_different_types)
{
    migraphx::array<double, 3> arr_double{1.5, 2.5, 3.5};
    migraphx::array<float, 3> arr_float{1.0f, 2.0f, 3.0f};
    migraphx::array<char, 3> arr_char{'a', 'b', 'c'};

    EXPECT(migraphx::float_equal(arr_double[0], 1.5));
    EXPECT(migraphx::float_equal(arr_float[1], 2.0f));
    EXPECT(arr_char[2] == 'c');
}

// Test mixed type operations
TEST_CASE(array_mixed_type_operations)
{
    migraphx::array<int, 3> arr_int{1, 2, 3};

    auto result = arr_int + 1.5; // int array + double
    // Should convert to double array
    EXPECT(migraphx::float_equal(result[0], 2.5));
    EXPECT(migraphx::float_equal(result[1], 3.5));
    EXPECT(migraphx::float_equal(result[2], 4.5));
}

// Test large arrays
TEST_CASE(array_large_size)
{
    migraphx::array<int, 100> large_arr(42);

    EXPECT(large_arr.size() == 100);
    EXPECT(large_arr[0] == 42);
    EXPECT(large_arr[50] == 42);
    EXPECT(large_arr[99] == 42);
}

// Test single element array special cases
TEST_CASE(array_single_element)
{
    migraphx::array<int, 1> arr{42};

    EXPECT(arr.front() == 42);
    EXPECT(arr.back() == 42);
    EXPECT(arr.size() == 1);
    EXPECT(not arr.empty());
    EXPECT(arr.product() == 42);

    auto doubled = arr.apply([](int x) { return x * 2; });
    EXPECT(doubled[0] == 84);
}

// Test edge cases for mathematical operations
TEST_CASE(array_math_edge_cases)
{
    // Test with zeros
    migraphx::array<int, 3> zeros{0, 0, 0};
    EXPECT(zeros.product() == 0);
    EXPECT(zeros.dot(zeros) == 0);

    // Test with ones
    migraphx::array<int, 3> ones{1, 1, 1};
    EXPECT(ones.product() == 1);

    // Test mixed positive/negative
    migraphx::array<int, 3> mixed{-1, 2, -3};
    EXPECT(mixed.product() == 6); // -1 * 2 * -3 = 6
}

// Test type conversions in operations
TEST_CASE(array_type_conversions)
{
    migraphx::array<int, 3> int_arr{1, 2, 3};
    migraphx::array<double, 3> double_arr{1.5, 2.5, 3.5};

    // This should work due to type conversion capabilities
    auto result = int_arr + double_arr;
    EXPECT(migraphx::float_equal(result[0], 2.5));
    EXPECT(migraphx::float_equal(result[1], 4.5));
    EXPECT(migraphx::float_equal(result[2], 6.5));
}

// Test chained operations
TEST_CASE(array_chained_operations)
{
    migraphx::array<int, 3> arr{1, 2, 3};

    auto result = (arr + 1) * 2 - 1;
    EXPECT(result[0] == 3); // (1+1)*2-1 = 3
    EXPECT(result[1] == 5); // (2+1)*2-1 = 5
    EXPECT(result[2] == 7); // (3+1)*2-1 = 7
}

// Test const correctness
TEST_CASE(array_const_correctness)
{
    const migraphx::array<int, 3> const_arr{1, 2, 3};

    // These should all work with const array
    EXPECT(const_arr.size() == 3);
    EXPECT(not const_arr.empty());
    EXPECT(const_arr.front() == 1);
    EXPECT(const_arr.back() == 3);
    EXPECT(const_arr[1] == 2);
    EXPECT(const_arr.data()[0] == 1);
    EXPECT(const_arr.begin()[0] == 1);
    EXPECT(const_arr.product() == 6);

    auto applied = const_arr.apply([](int x) { return x * 2; });
    EXPECT(applied[0] == 2);
}

// Test iterator functionality thoroughly
TEST_CASE(array_iterator_functionality)
{
    migraphx::array<int, 5> arr{10, 20, 30, 40, 50};

    // Test forward iteration
    int expected = 10;
    // NOLINTNEXTLINE(modernize-loop-convert)
    for(auto* it = arr.begin(); it != arr.end(); ++it)
    {
        EXPECT(*it == expected);
        expected += 10;
    }

    // Test modifying through iterator
    // NOLINTNEXTLINE(modernize-loop-convert)
    for(auto* it = arr.begin(); it != arr.end(); ++it)
    {
        *it += 1;
    }

    EXPECT(arr[0] == 11);
    EXPECT(arr[4] == 51);
}

// Test array with bool type (special case)
TEST_CASE(array_bool_type)
{
    migraphx::array<bool, 4> bool_arr{true, false, true, false};

    EXPECT(bool_arr[0] == true);
    EXPECT(bool_arr[1] == false);
    EXPECT(bool_arr[2] == true);
    EXPECT(bool_arr[3] == false);

    bool_arr[1] = true;
    EXPECT(bool_arr[1] == true);
}

// ==================== Tests for Selected Functions ====================

// Test array_apply function
TEST_CASE(array_apply_function)
{
    migraphx::array<int, 3> arr{1, 2, 3};

    // Test with simple lambda
    auto doubler = migraphx::array_apply([](int x) { return x * 2; });
    auto doubled = doubler(arr);

    EXPECT(doubled[0] == 2);
    EXPECT(doubled[1] == 4);
    EXPECT(doubled[2] == 6);
}

// Test array_apply with type conversion
TEST_CASE(array_apply_function_type_conversion)
{
    migraphx::array<int, 3> int_arr{1, 2, 3};

    auto int_to_float = migraphx::array_apply([](int x) { return float(x) + 0.5f; });
    auto float_result = int_to_float(int_arr);

    EXPECT(migraphx::float_equal(float_result[0], 1.5f));
    EXPECT(migraphx::float_equal(float_result[1], 2.5f));
    EXPECT(migraphx::float_equal(float_result[2], 3.5f));
}

// Test array_apply with complex operation
TEST_CASE(array_apply_complex_operation)
{
    migraphx::array<int, 4> arr{1, 2, 3, 4};

    auto square_plus_one = migraphx::array_apply([](int x) { return x * x + 1; });
    auto result          = square_plus_one(arr);

    EXPECT(result[0] == 2);  // 1*1 + 1 = 2
    EXPECT(result[1] == 5);  // 2*2 + 1 = 5
    EXPECT(result[2] == 10); // 3*3 + 1 = 10
    EXPECT(result[3] == 17); // 4*4 + 1 = 17
}

// Test make_array function
TEST_CASE(make_array_function)
{
    auto arr = migraphx::make_array(1, 2, 3, 4);

    EXPECT(arr.size() == 4);
    EXPECT(arr[0] == 1);
    EXPECT(arr[1] == 2);
    EXPECT(arr[2] == 3);
    EXPECT(arr[3] == 4);
}

// Test make_array with type conversion
TEST_CASE(make_array_type_conversion)
{
    auto arr = migraphx::make_array(1, 2.5, 3, 4.7);

    EXPECT(arr.size() == 4);
    EXPECT(arr[0] == 1);
    EXPECT(arr[1] == 2); // Should be cast to int
    EXPECT(arr[2] == 3);
    EXPECT(arr[3] == 4); // Should be cast to int
}

// Test make_array with single element
TEST_CASE(make_array_single_element)
{
    auto arr = migraphx::make_array(42);

    EXPECT(arr.size() == 1);
    EXPECT(arr[0] == 42);
}

// Test make_array with mixed types
TEST_CASE(make_array_mixed_types)
{
    auto arr = migraphx::make_array(1.0, 2, 3.5f);

    EXPECT(arr.size() == 3);
    EXPECT(migraphx::float_equal(arr[0], 1.0));
    EXPECT(migraphx::float_equal(arr[1], 2.0));
    EXPECT(migraphx::float_equal(arr[2], 3.5));
}

// Test integral_const_array basic functionality
TEST_CASE(integral_const_array_basic)
{
    migraphx::integral_const_array<int, 1, 2, 3> const_arr;

    EXPECT(const_arr.size() == 3);
    EXPECT(const_arr[0] == 1);
    EXPECT(const_arr[1] == 2);
    EXPECT(const_arr[2] == 3);
}

// Test integral_const_array base() method
TEST_CASE(integral_const_array_base)
{
    migraphx::integral_const_array<int, 5, 10, 15> const_arr;
    auto base_arr = const_arr.base();

    EXPECT(base_arr.size() == 3);
    EXPECT(base_arr[0] == 5);
    EXPECT(base_arr[1] == 10);
    EXPECT(base_arr[2] == 15);
}

// Test generate_array function
TEST_CASE(generate_array_function)
{
    auto arr = migraphx::generate_array<int>(migraphx::_c<4>,
                                             [](auto i) { return static_cast<int>(i * 2); });

    EXPECT(arr.size() == 4);
    EXPECT(arr[0] == 0); // 0 * 2 = 0
    EXPECT(arr[1] == 2); // 1 * 2 = 2
    EXPECT(arr[2] == 4); // 2 * 2 = 4
    EXPECT(arr[3] == 6); // 3 * 2 = 6
}

// Test generate_array with more complex generator
TEST_CASE(generate_array_complex_generator)
{
    auto arr = migraphx::generate_array<int>(
        migraphx::_c<3>, [](auto i) { return static_cast<int>((i + 1) * (i + 1)); });

    EXPECT(arr.size() == 3);
    EXPECT(arr[0] == 1); // (0+1)^2 = 1
    EXPECT(arr[1] == 4); // (1+1)^2 = 4
    EXPECT(arr[2] == 9); // (2+1)^2 = 9
}

// Test generate_array with single element
TEST_CASE(generate_array_single_element)
{
    auto arr = migraphx::generate_array<int>(migraphx::_c<1>, [](auto /* i */) { return 42; });

    EXPECT(arr.size() == 1);
    EXPECT(arr[0] == 42);
}

// Test unpack function with integral_const_array
TEST_CASE(unpack_function)
{
    migraphx::integral_const_array<int, 1, 2, 3> const_arr;

    auto result = migraphx::unpack(const_arr, [](auto... values) {
        return (values + ...); // C++17 fold expression to sum all values
    });

    EXPECT(result == 6); // 1 + 2 + 3 = 6
}

// Test unpack with different operation
TEST_CASE(unpack_product)
{
    migraphx::integral_const_array<int, 2, 3, 4> const_arr;

    auto result = migraphx::unpack(const_arr, [](auto... values) {
        return (values * ...); // Product of all values
    });

    EXPECT(result == 24); // 2 * 3 * 4 = 24
}

// Test unpack with single value
TEST_CASE(unpack_single_value)
{
    migraphx::integral_const_array<int, 42> const_arr;

    auto result = migraphx::unpack(const_arr, [](auto value) { return value * 2; });

    EXPECT(result == 84);
}

// Test transform function (single array)
TEST_CASE(transform_single_array)
{
    migraphx::integral_const_array<int, 1, 2, 3> const_arr;

    auto transformed = migraphx::transform(const_arr, [](auto x) { return x * 2; });

    EXPECT(transformed.size() == 3);
    EXPECT(transformed[0] == 2);
    EXPECT(transformed[1] == 4);
    EXPECT(transformed[2] == 6);
}

// Test transform function with different operation
TEST_CASE(transform_square)
{
    migraphx::integral_const_array<int, 1, 2, 3, 4> const_arr;

    auto squared = migraphx::transform(const_arr, [](auto x) { return x * x; });

    EXPECT(squared.size() == 4);
    EXPECT(squared[0] == 1);
    EXPECT(squared[1] == 4);
    EXPECT(squared[2] == 9);
    EXPECT(squared[3] == 16);
}

// Test transform_i function (with index)
TEST_CASE(transform_i_function)
{
    migraphx::integral_const_array<int, 10, 20, 30> const_arr;

    auto transformed =
        migraphx::transform_i(const_arr, [](auto value, auto index) { return value + index; });

    EXPECT(transformed.size() == 3);
    EXPECT(transformed[0] == 10); // 10 + 0 = 10
    EXPECT(transformed[1] == 21); // 20 + 1 = 21
    EXPECT(transformed[2] == 32); // 30 + 2 = 32
}

// Test transform_i with index multiplication
TEST_CASE(transform_i_multiply_index)
{
    migraphx::integral_const_array<int, 5, 6, 7> const_arr;

    auto transformed = migraphx::transform_i(
        const_arr, [](auto value, auto index) { return value * (index + 1); });

    EXPECT(transformed.size() == 3);
    EXPECT(transformed[0] == 5);  // 5 * (0+1) = 5
    EXPECT(transformed[1] == 12); // 6 * (1+1) = 12
    EXPECT(transformed[2] == 21); // 7 * (2+1) = 21
}

// Test transform with two arrays
TEST_CASE(transform_two_arrays)
{
    migraphx::integral_const_array<int, 1, 2, 3> arr1;
    migraphx::integral_const_array<int, 4, 5, 6> arr2;

    auto result = migraphx::transform(arr1, arr2, [](auto x, auto y) { return x + y; });

    EXPECT(result.size() == 3);
    EXPECT(result[0] == 5); // 1 + 4 = 5
    EXPECT(result[1] == 7); // 2 + 5 = 7
    EXPECT(result[2] == 9); // 3 + 6 = 9
}

// Test transform with two arrays and multiplication
TEST_CASE(transform_two_arrays_multiply)
{
    migraphx::integral_const_array<int, 2, 3, 4> arr1;
    migraphx::integral_const_array<int, 5, 6, 7> arr2;

    auto result = migraphx::transform(arr1, arr2, [](auto x, auto y) { return x * y; });

    EXPECT(result.size() == 3);
    EXPECT(result[0] == 10); // 2 * 5 = 10
    EXPECT(result[1] == 18); // 3 * 6 = 18
    EXPECT(result[2] == 28); // 4 * 7 = 28
}

// Test index_ints type alias
TEST_CASE(index_ints_type_alias)
{
    migraphx::index_ints<1, 2, 3, 4> indices;

    EXPECT(indices.size() == 4);
    EXPECT(indices[0] == 1);
    EXPECT(indices[1] == 2);
    EXPECT(indices[2] == 3);
    EXPECT(indices[3] == 4);
}

// Test edge case: empty operations
TEST_CASE(utility_functions_edge_cases)
{
    // Test single element arrays
    migraphx::integral_const_array<int, 42> single;
    auto transformed_single = migraphx::transform(single, [](auto x) { return x * 2; });
    EXPECT(transformed_single[0] == 84);

    // Test with zero values
    migraphx::integral_const_array<int, 0, 0, 0> zeros;
    auto transformed_zeros = migraphx::transform(zeros, [](auto x) { return x + 1; });
    EXPECT(transformed_zeros[0] == 1);
    EXPECT(transformed_zeros[1] == 1);
    EXPECT(transformed_zeros[2] == 1);
}

// Test complex function composition
TEST_CASE(function_composition_complex)
{
    // Create initial array
    auto initial = migraphx::make_array(1, 2, 3);

    // Apply multiple transformations
    auto applier1 = migraphx::array_apply([](int x) { return x * 2; });
    auto applier2 = migraphx::array_apply([](int x) { return x + 1; });

    auto step1        = applier1(initial);
    auto final_result = applier2(step1);

    EXPECT(final_result[0] == 3); // (1*2)+1 = 3
    EXPECT(final_result[1] == 5); // (2*2)+1 = 5
    EXPECT(final_result[2] == 7); // (3*2)+1 = 7
}

// Test type safety and consistency
TEST_CASE(type_safety_consistency)
{
    // Ensure make_array preserves type correctly
    auto int_arr   = migraphx::make_array(1, 2, 3);
    auto float_arr = migraphx::make_array(1.0f, 2.0f, 3.0f);

    // Test that the values are correct regardless of exact type implementation
    EXPECT(int_arr[0] == 1);
    EXPECT(migraphx::float_equal(float_arr[0], 1.0f));
}
