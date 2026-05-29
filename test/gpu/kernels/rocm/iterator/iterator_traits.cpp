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
#include <rocm/iterator/iterator_traits.hpp>
#include <migraphx/kernels/test.hpp>

// ---- Iterator category tag hierarchy ----

TEST_CASE(tag_hierarchy)
{
    EXPECT(rocm::is_base_of<rocm::input_iterator_tag, rocm::forward_iterator_tag>{});
    EXPECT(rocm::is_base_of<rocm::forward_iterator_tag, rocm::bidirectional_iterator_tag>{});
    EXPECT(rocm::is_base_of<rocm::bidirectional_iterator_tag, rocm::random_access_iterator_tag>{});
    EXPECT(rocm::is_base_of<rocm::input_iterator_tag, rocm::random_access_iterator_tag>{});
}

TEST_CASE(tag_not_related)
{
    EXPECT(not rocm::is_base_of<rocm::output_iterator_tag, rocm::input_iterator_tag>{});
    EXPECT(not rocm::is_base_of<rocm::input_iterator_tag, rocm::output_iterator_tag>{});
    EXPECT(not rocm::is_base_of<rocm::random_access_iterator_tag, rocm::input_iterator_tag>{});
}

// ---- Pointer specialization: non-const ----

TEST_CASE(pointer_difference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<int*>::difference_type, rocm::ptrdiff_t>{});
}

TEST_CASE(pointer_value_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<int*>::value_type, int>{});
}

TEST_CASE(pointer_pointer_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<int*>::pointer, int*>{});
}

TEST_CASE(pointer_reference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<int*>::reference, int&>{});
}

TEST_CASE(pointer_iterator_category)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<int*>::iterator_category,
                         rocm::random_access_iterator_tag>{});
}

// ---- Pointer specialization: const ----

TEST_CASE(const_pointer_value_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<const int*>::value_type, int>{});
}

TEST_CASE(const_pointer_pointer_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<const int*>::pointer, const int*>{});
}

TEST_CASE(const_pointer_reference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<const int*>::reference, const int&>{});
}

TEST_CASE(const_pointer_difference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<const int*>::difference_type, rocm::ptrdiff_t>{});
}

TEST_CASE(const_pointer_iterator_category)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<const int*>::iterator_category,
                         rocm::random_access_iterator_tag>{});
}

// ---- Pointer specialization: volatile ----

TEST_CASE(volatile_pointer_value_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<volatile int*>::value_type, int>{});
}

TEST_CASE(volatile_pointer_pointer_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<volatile int*>::pointer, volatile int*>{});
}

TEST_CASE(volatile_pointer_reference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<volatile int*>::reference, volatile int&>{});
}

// ---- Pointer specialization: const volatile ----

TEST_CASE(const_volatile_pointer_value_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<const volatile int*>::value_type, int>{});
}

// ---- Pointer specialization: other types ----

TEST_CASE(float_pointer_value_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<float*>::value_type, float>{});
}

TEST_CASE(double_pointer_difference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<double*>::difference_type, rocm::ptrdiff_t>{});
}

TEST_CASE(char_pointer_reference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<char*>::reference, char&>{});
}

// ---- Primary template with custom iterator ----

struct mock_iterator
{
    using difference_type   = int;
    using value_type        = double;
    using pointer           = double*;
    using reference         = double&;
    using iterator_category = rocm::bidirectional_iterator_tag;
};

TEST_CASE(custom_iterator_difference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<mock_iterator>::difference_type, int>{});
}

TEST_CASE(custom_iterator_value_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<mock_iterator>::value_type, double>{});
}

TEST_CASE(custom_iterator_pointer_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<mock_iterator>::pointer, double*>{});
}

TEST_CASE(custom_iterator_reference_type)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<mock_iterator>::reference, double&>{});
}

TEST_CASE(custom_iterator_category)
{
    EXPECT(rocm::is_same<rocm::iterator_traits<mock_iterator>::iterator_category,
                         rocm::bidirectional_iterator_tag>{});
}
