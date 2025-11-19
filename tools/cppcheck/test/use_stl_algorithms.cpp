// Test for useStlAlgorithm rules from rules.xml
#include <cstring>

void test_memcpy_usage()
{
    char src[] = "hello";
    char dest[10];
    // cppcheck-suppress migraphx-useStlAlgorithms
    memcpy(dest, src, 5);
}

void test_strcpy_usage()
{
    char src[] = "hello";
    char dest[10];
    // cppcheck-suppress migraphx-useStlAlgorithms
    strcpy(dest, src);
}

void test_strncpy_usage()
{
    char src[] = "hello";
    char dest[10];
    // cppcheck-suppress migraphx-useStlAlgorithms
    strncpy(dest, src, 5);
}

void test_memset_usage()
{
    char dest[10];
    // cppcheck-suppress migraphx-useStlAlgorithms
    memset(dest, 0, 10);
}

void test_memcmp_usage()
{
    char src[]    = "hello";
    char dest[10] = "hello";
    // cppcheck-suppress migraphx-useStlAlgorithms
    int result = memcmp(src, dest, 5);
    (void)result; // Suppress unused variable warning
}

void test_memchr_usage()
{
    char src[] = "hello";
    // cppcheck-suppress migraphx-useStlAlgorithms
    void* found = memchr(src, 'l', 5);
    (void)found; // Suppress unused variable warning
}

void test_strcat_usage()
{
    char dest[20] = "hello";
    char src[]    = " world";
    // cppcheck-suppress migraphx-useStlAlgorithms
    strcat(dest, src);
}

void test_strncat_usage()
{
    char dest[20] = "hello";
    char src[]    = " world";
    // cppcheck-suppress migraphx-useStlAlgorithms
    strncat(dest, src, 3);
}

void test_arithmetic_should_not_trigger()
{
    // Should not trigger: simple arithmetic
    int x   = 5;
    int y   = 10;
    int sum = x + y;
    (void)sum; // Suppress unused variable warning
}

void test_array_access_should_not_trigger()
{
    // Should not trigger: direct array access
    char arr[] = "hello";
    char c     = arr[0];
    (void)c; // Suppress unused variable warning
}
