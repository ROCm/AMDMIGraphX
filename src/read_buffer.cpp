#include <migraphx/read_buffer.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<char> read_buffer(const std::string& filename)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if(not is.good())
        MIGRAPHX_THROW("Cannot open file: " + filename);
    std::streamsize size = is.tellg();
    is.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if(!is.read(buffer.data(), size))
    {
        MIGRAPHX_THROW("Error reading file: " + filename);
    }
    return buffer;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
