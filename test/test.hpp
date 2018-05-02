
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef RTG_GUARD_TEST_TEST_HPP
#define RTG_GUARD_TEST_TEST_HPP

inline void failed(bool b, const char* msg, const char* file, int line)
{
    if (!b)
        std::cout << "FAILED: " << msg << ": " << file << ": " << line << std::endl;
}

inline void failed_abort(bool b, const char* msg, const char* file, int line)
{
    if (!b) 
    {
        std::cout << "FAILED: " << msg << ": " << file << ": " << line << std::endl;
        std::abort();
    }
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

// NOLINTNEXTLINE
#define CHECK(...) failed(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__)
// NOLINTNEXTLINE
#define EXPECT(...) failed_abort(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__)
// NOLINTNEXTLINE
#define EXPECT_EQUAL(LEFT, RIGHT) expect_equality(LEFT, RIGHT, #LEFT, #RIGHT, __FILE__, __LINE__)
// NOLINTNEXTLINE
#define STATUS(...) EXPECT((__VA_ARGS__) == 0)

// NOLINTNEXTLINE
#define FAIL(...) failed(false, __VA_ARGS__, __FILE__, __LINE__)

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
