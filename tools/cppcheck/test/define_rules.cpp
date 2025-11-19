// Test for defineUpperCase and definePrefix rules from rules.xml

// cppcheck-suppress defineUpperCase
#define myMacro 42

// cppcheck-suppress defineUpperCase
#define TestMacro 100

// cppcheck-suppress definePrefix
#define MY_CONSTANT 5

// cppcheck-suppress definePrefix
#define OTHER_PREFIX_VALUE 10

// cppcheck-suppress defineUpperCase
// cppcheck-suppress definePrefix
#define badMacro 15

#define MIGRAPHX_CORRECT_MACRO 20
#define MIGRAPHX_ANOTHER_MACRO 30

void test_macro_not_uppercase_1() { int x = myMacro; }

void test_macro_not_uppercase_2() { int y = TestMacro; }

void test_macro_missing_prefix_1() { int z = MY_CONSTANT; }

void test_macro_missing_prefix_2() { int w = OTHER_PREFIX_VALUE; }

void test_macro_both_issues() { int v = badMacro; }

void test_correct_macros()
{
    int correct = MIGRAPHX_CORRECT_MACRO;
    int another = MIGRAPHX_ANOTHER_MACRO;
}

void test_standard_library_macros()
{
#ifndef NULL
#define NULL nullptr
#endif
}
