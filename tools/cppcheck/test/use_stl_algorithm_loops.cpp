#include <algorithm>
#include <cmath>
#include <cstdlib>

// Test for useStlAlgorithm rules from rules.xml (for loop patterns)

void test_for_loop_std_fill_pattern()
{
    int arr[10];
    // cppcheck-suppress useStlAlgorithm
    for(int i = 0; i < 10; i++)
    {
        arr[i] = 5;
    }
}

void test_for_loop_std_generate_pattern()
{
    int arr[10];
    // cppcheck-suppress useStlAlgorithm
    for(int i = 0; i < 10; i++)
    {
        arr[i] = rand();
    }
}

void test_for_loop_std_transform_unary_pattern()
{
    int arr[10];
    int other_arr[10] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    // cppcheck-suppress useStlAlgorithm
    for(int i = 0; i < 10; i++)
    {
        arr[i] = abs(other_arr[i]);
    }
}

void test_for_loop_std_transform_binary_pattern()
{
    int arr[10]       = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int other_arr[10] = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9};
    // cppcheck-suppress useStlAlgorithm
    for(int i = 0; i < 10; i++)
    {
        arr[i] = std::max(other_arr[i], arr[i]);
    }
}

void test_complex_loop_should_not_trigger()
{
    // Should not trigger: complex logic
    int arr[10];
    for(int i = 0; i < 10; i++)
    {
        if(i % 2 == 0)
        {
            arr[i] = 5;
        }
        else
        {
            arr[i] = 10;
        }
    }
}

void test_multiple_operations_should_not_trigger()
{
    // Should not trigger: multiple operations
    int arr[10];
    int other_arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for(int i = 0; i < 10; i++)
    {
        arr[i]       = other_arr[i];
        other_arr[i] = 0;
    }
}

void test_complex_pattern_should_not_trigger()
{
    // Should not trigger: not a simple pattern
    int arr[10]       = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int other_arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for(int i = 1; i < 10; i++)
    {
        arr[i] = arr[i - 1] + other_arr[i];
    }
}

// Mock functions for compilation
int rand() { return 0; }
int abs(int x) { return x > 0 ? x : -x; }
int max(int a, int b) { return a > b ? a : b; }
