// Test for EmptyCatchStatement check
#include <iostream>
#include <stdexcept>

void test_empty_catch()
{
    try
    {
        throw std::runtime_error("test");
    }
    // cppcheck-suppress migraphx-EmptyCatchStatement
    catch(const std::exception& e)
    {
    }
}

void test_empty_catch_ellipsis()
{
    try
    {
        throw 42;
    }
    // cppcheck-suppress migraphx-EmptyCatchStatement
    catch(...)
    {
    }
}

void test_catch_with_statement()
{
    try
    {
        throw std::runtime_error("test");
    }
    catch(const std::exception& e)
    {
        // This is acceptable
        return;
    }
}

void test_catch_with_logging()
{
    try
    {
        throw std::runtime_error("test");
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
