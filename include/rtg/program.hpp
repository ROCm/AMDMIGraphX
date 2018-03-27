#ifndef GUARD_RTGLIB_PROGRAM_HPP
#define GUARD_RTGLIB_PROGRAM_HPP

#include <deque>
#include <unordered_map>
#include <rtg/instruction.hpp>

namespace rtg {

struct program
{
    // A deque is used to keep references to an instruction stable
    std::deque<instruction> instructions;

    std::unordered_map<std::string, operand> ops;

};

}

#endif
