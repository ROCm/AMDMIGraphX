// Test for LambdaAttribute check

void test_device_attribute_before_params()
{
    int x = 5;
    // cppcheck-suppress migraphx-LambdaAttribute
    auto lambda1 = [] __device__(int a) { return a * 2; };
}

void test_host_attribute_before_params()
{
    // cppcheck-suppress migraphx-LambdaAttribute
    auto lambda2 = [] __host__(int a) { return a + 1; };
}

void test_device_attribute_before_brace()
{
    // cppcheck-suppress migraphx-LambdaAttribute
    auto lambda3 = [] __device__ { return 42; };
}

void test_attribute_after_params()
{
    auto lambda1 = [](int a) __device__ { return a * 2; };
}

void test_no_attributes()
{
    auto lambda2 = [](int a) { return a + 1; };
}

void test_capture_list_only()
{
    int x        = 5;
    auto lambda3 = [x](int a) { return a + x; };
}

void test_attribute_in_correct_position()
{
    auto lambda4 = [](int a) __host__ __device__ { return a; };
}
