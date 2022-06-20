#include <migraphx/file_buffer.hpp>
#include <migraphx/errors.hpp>
#include <fstream>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
T generic_read_file(const std::string& filename)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = is.tellg();
    if(size < 1)
        MIGRAPHX_THROW("Invalid size for: " + filename);
    is.seekg(0, std::ios::beg);

    T buffer(size, 0);
    if(!is.read(&buffer[0], size))
        MIGRAPHX_THROW("Error reading file: " + filename);
    return buffer;
}

std::vector<char> read_buffer(const std::string& filename)
{
    return generic_read_file<std::vector<char>>(filename);
}

std::string read_string(const std::string& filename)
{
    return generic_read_file<std::string>(filename);
}

void write_buffer(const std::string& filename, const char* buffer, std::size_t size)
{
    std::ofstream os(filename);
    os.write(buffer, size);
}
void write_buffer(const std::string& filename, const std::vector<char>& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
