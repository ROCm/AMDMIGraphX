#include <string>

// Test for UseSmartPointer check

void test_positive_cases()
{
    // Should trigger: new usage without smart pointer
    // cppcheck-suppress migraphx-UseSmartPointer
    int* ptr1 = new int(5);

    // Should trigger: new array
    // cppcheck-suppress migraphx-UseSmartPointer
    int* ptr2 = new int[10];

    // Should trigger: new with class
    // cppcheck-suppress migraphx-UseSmartPointer
    std::string* str_ptr = new std::string("hello");

    // Should trigger: new with user-defined type
    class MyClass
    {
    };
    // cppcheck-suppress migraphx-UseSmartPointer
    MyClass* obj_ptr = new MyClass();

    // Clean up to avoid memory leaks
    delete ptr1;
    delete[] ptr2;
    delete str_ptr;
    delete obj_ptr;
}

void test_negative_cases()
{
    // Should not trigger: using smart pointers
    // std::unique_ptr<int> smart_ptr = std::make_unique<int>(5);
    // std::shared_ptr<int> shared_ptr = std::make_shared<int>(10);

    // Should not trigger: stack allocation
    int local_var = 5;
    int array[10];
    (void)local_var; // Use variables to avoid warnings
    (void)array;

    // Should not trigger: function parameters
    // void func(int* ptr);

    // Should not trigger: member variables
    class TestClass
    {
        int* member_ptr; // This might be managed elsewhere
    };

    // Should not trigger: placement new (advanced case)
    // char buffer[sizeof(int)];
    // int* ptr = new(buffer) int(42);
}
