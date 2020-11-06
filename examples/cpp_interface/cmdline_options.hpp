#ifndef __CMDLINE_OPTIONS_HPP__
#define __CMDLINE_OPTIONS_HPP__
#include <string>

bool cmdOptionExists(char** begin, char** end, const std::string& option);
char* getCmdOption(char** begin, char** end, const std::string& option);

#endif
