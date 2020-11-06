#include "cmdline_options.hpp"

#include <algorithm>

char* getCmdOption(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if(itr != end && ++itr != end)
    {
        return *itr;
    }

    return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}
