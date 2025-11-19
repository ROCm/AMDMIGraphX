// Test for StrlenEmptyString rule from rules.xml
#include <cstring>

void test_strlen_greater_than_zero()
{
    char str[] = "hello";
    // cppcheck-suppress StrlenEmptyString
    if(strlen(str) > 0)
    {
        // String is not empty
    }
}

void test_strlen_empty_string_check()
{
    char str[] = "";
    // cppcheck-suppress StrlenEmptyString
    if(strlen(str) > 0)
    {
        // String is not empty
    }
}

void test_strlen_negated_condition()
{
    char str[] = "hello";
    // cppcheck-suppress StrlenEmptyString
    if(!strlen(str))
    {
        // String is empty
    }
}

void test_strlen_specific_length_should_not_trigger()
{
    // Should not trigger: checking actual length, not emptiness
    char str[] = "hello";
    if(strlen(str) == 5)
    {
        // String has specific length
    }
}

void test_strlen_for_assignment_should_not_trigger()
{
    // Should not trigger: using length for other purposes
    char str[] = "hello";
    int len    = strlen(str);
    (void)len; // Suppress unused variable warning
}

void test_direct_empty_check_should_not_trigger()
{
    // Should not trigger: direct empty check without strlen
    char str[] = "hello";
    if(str[0] == '\0')
    {
        // String is empty
    }
}

void test_strcmp_should_not_trigger()
{
    // Should not trigger: comparing strings, not checking emptiness
    char str[] = "hello";
    if(strcmp(str, "hello") == 0)
    {
        // Strings are equal
    }
}

// Mock strlen and strcmp for compilation
size_t strlen(const char*) { return 0; }
int strcmp(const char*, const char*) { return 0; }
