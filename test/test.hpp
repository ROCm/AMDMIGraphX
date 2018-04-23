
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef GUARD_TEST_TEST_HPP
#define GUARD_TEST_TEST_HPP

inline void failed(const char* msg, const char* file, int line)
{
    std::cout << "FAILED: " << msg << ": " << file << ": " << line << std::endl;
}

[[gnu::noreturn]] inline void failed_abort(const char* msg, const char* file, int line)
{
    failed(msg, file, line);
    std::abort();
}

template <class TLeft, class TRight>
inline void expect_equality(const TLeft& left,
                            const TRight& right,
                            const char* left_s,
                            const char* riglt_s,
                            const char* file,
                            int line)
{
    if(left == right)
        return;

    std::cout << "FAILED: " << left_s << "(" << left << ") == " << riglt_s << "(" << right
              << "): " << file << ':' << line << std::endl;
    std::abort();
}

#define CHECK(...)     \
    if(!(__VA_ARGS__)) \
    failed(#__VA_ARGS__, __FILE__, __LINE__)
#define EXPECT(...)    \
    if(!(__VA_ARGS__)) \
    failed_abort(#__VA_ARGS__, __FILE__, __LINE__)
#define EXPECT_EQUAL(LEFT, RIGHT) expect_equality(LEFT, RIGHT, #LEFT, #RIGHT, __FILE__, __LINE__)
#define STATUS(...) EXPECT((__VA_ARGS__) == 0)

#define FAIL(...) failed(__VA_ARGS__, __FILE__, __LINE__)

template <class F>
bool throws(F f)
{
    try
    {
        f();
        return false;
    }
    catch(...)
    {
        return true;
    }
}

template <class F, class Exception>
bool throws(F f, std::string msg = "")
{
    try
    {
        f();
        return false;
    }
    catch(const Exception& ex)
    {
        return std::string(ex.what()).find(msg) != std::string::npos;
    }
}

template <class T>
void run_test()
{
    T t = {};
    t.run();
}

#endif
