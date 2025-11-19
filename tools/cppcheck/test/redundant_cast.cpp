// Test for RedundantCast check

void test_redundant_auto_cast_1() {
    // cppcheck-suppress migraphx-RedundantCast
    auto x = static_cast<int>(5);
}

void test_redundant_auto_cast_2() {
    int y = 10;
    // cppcheck-suppress migraphx-RedundantCast
    auto z = static_cast<int>(y);
}

void test_redundant_auto_cast_3() {
    // cppcheck-suppress migraphx-RedundantCast
    auto w = static_cast<double>(3.14);
}

void test_explicit_type_cast() {
    int x = static_cast<int>(5.5);
}

void test_cast_to_different_type() {
    int y = 10;
    double z = static_cast<double>(y);
}

void test_cast_with_const() {
    const int a = 5;
    auto b = static_cast<const int&>(a);
}

void test_no_cast() {
    auto c = 42;
    int d = 5;
}
