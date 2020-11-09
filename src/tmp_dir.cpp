#include <migraphx/tmp_dir.hpp>
#include <migraphx/env.hpp>
#include <migraphx/errors.hpp>
#include <algorithm>
#include <random>
#include <thread>
#include <sstream>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <unistd.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DEBUG_SAVE_TEMP_DIR)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_CMD_EXECUTE)

std::string random_string(std::string::size_type length)
{
    static const std::string& chars = "0123456789"
                                      "abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::mt19937 rg{std::random_device{}()};
    std::uniform_int_distribution<std::string::size_type> pick(0, chars.length() - 1);

    std::string str(length, 0);
    std::generate(str.begin(), str.end(), [&] { return chars[pick(rg)]; });

    return str;
}

std::string unique_string(const std::string& prefix)
{
    auto pid = getpid();
    auto tid = std::this_thread::get_id();
    std::stringstream ss;
    ss << prefix << "-" << pid << "-" << tid << "-" << random_string(64);
    return ss.str();
}

tmp_dir::tmp_dir() : path(fs::temp_directory_path() / unique_string("migraphx"))
{
    fs::create_directories(this->path);
}

void system_cmd(const std::string& cmd)
{
// We shouldn't call system commands
#ifdef MIGRAPHX_USE_CLANG_TIDY
    (void)cmd;
#else
    if(std::system(cmd.c_str()) != 0)
        MIGRAPHX_THROW("Can't execute " + cmd);
#endif
}

void tmp_dir::execute(const std::string& exe, const std::string& args) const
{
    std::string cd  = "cd " + this->path.string() + "; ";
    std::string cmd = cd + exe + " " + args; // + " > /dev/null";
    if(enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
        std::cout << cmd << std::endl;
    system_cmd(cmd);
}

tmp_dir::~tmp_dir()
{
    if(!enabled(MIGRAPHX_DEBUG_SAVE_TEMP_DIR{}))
    {
        fs::remove_all(this->path);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
