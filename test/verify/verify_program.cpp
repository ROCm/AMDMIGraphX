#include "auto_register_verify_program.hpp"

std::vector<program_info>& get_programs_vector()
{
    static std::vector<program_info> result;
    return result;
}

void register_program_info(const program_info& pi)
{
    get_programs_vector().push_back(pi);
}
const std::vector<program_info>& get_programs()
{
    return get_programs_vector();
}
