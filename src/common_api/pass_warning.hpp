#ifndef PASS_WARNING_HPP
#define PASS_WARNING_HPP

#include <iostream>

inline void
pass_impl(const std::string& msg, const char* file, const char* fun, int line, bool term)
{
    std::cout << file << ":" << line << ":" << fun << ": " << msg << std::endl;
    if(term)
        std::terminate();
}

#define pass_warning(msg, term) pass_impl(msg, __FILE__, __FUNCTION__, __LINE__, term)

#endif // PASS_WARNING_HPP
