// Test for ConditionalAssert check
#include <cassert>

void test_redundant_if_before_assert() {
    int x = 5;
    // cppcheck-suppress migraphx-ConditionalAssert
    if (x > 0) {
        assert(x > 0);
    }
}

void test_redundant_if_before_assert_different_condition() {
    int x = 5;
    // cppcheck-suppress migraphx-ConditionalAssert
    if (x != 0) {
        assert(x != 0);
    }
}

void test_different_conditions() {
    int x = 5;
    if (x > 0) {
        assert(x < 10);
    }
}

void test_assert_without_if() {
    int x = 5;
    assert(x > 0);
}

void test_if_without_assert() {
    int x = 5;
    if (x > 0) {
        x = x + 1;
    }
}

void test_multiple_statements_in_if() {
    int x = 5;
    if (x > 0) {
        int y = x * 2;
        assert(x > 0);
    }
}
