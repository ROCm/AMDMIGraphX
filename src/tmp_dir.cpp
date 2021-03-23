#include <migraphx/tmp_dir.hpp>
#include <migraphx/env.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/process.hpp>
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
    auto clk = std::chrono::steady_clock::now().time_since_epoch().count();
    std::stringstream ss;
    ss << std::hex << prefix << "-" << pid << "-" << tid << "-" << clk << "-" << random_string(16);
    return ss.str();
}

tmp_dir::tmp_dir(const std::string& prefix)
    : path(fs::temp_directory_path() /
           unique_string(prefix.empty() ? "migraphx" : "migraphx-" + prefix))
{
    fs::create_directories(this->path);
}

void tmp_dir::execute(const std::string& exe, const std::string& args) const
{
    process{exe + " " + args}.cwd(this->path).exec();
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
