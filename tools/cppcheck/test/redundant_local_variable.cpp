// Test for RedundantLocalVariable check

int test_positive_cases() {
    int x = 5;
    
    // Should trigger: variable returned immediately after declaration
    // cppcheck-suppress migraphx-RedundantLocalVariable
    int result = x * 2;
    return result;
}

int test_positive_case2(int a, int b) {
    // Should trigger: complex expression assigned and returned
    // cppcheck-suppress migraphx-RedundantLocalVariable
    int value = a + b * 2;
    return value;
}

int test_negative_cases(int x) {
    // Should not trigger: variable used before return
    int result = x * 2;
    result = result + 1;
    return result;
}

int test_negative_case2(int x) {
    // Should not trigger: multiple statements between declaration and return
    int result = x * 2;
    int temp = result + 1;
    return result;
}

int test_negative_case3(int x) {
    // Should not trigger: no return statement
    int result = x * 2;
    result = result + 1;
    return 0;
}

void test_negative_case4(int x) {
    // Should not trigger: void function
    int result = x * 2;
    return;
}
