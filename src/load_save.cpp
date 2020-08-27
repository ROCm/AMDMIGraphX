#include <migraphx/load_save.hpp>
#include <migraphx/json.hpp>
#include <migraphx/msgpack.hpp>
#include <fstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<char> read_buffer(const std::string& filename)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = is.tellg();
    is.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if(!is.read(buffer.data(), size))
    {
        MIGRAPHX_THROW("Error reading file: " + filename);
    }
    return buffer;
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

program load(const std::string& filename, const file_options& options)
{
    return load_buffer(read_buffer(filename), options);
}
program load_buffer(const std::vector<char>& buffer, const file_options& options)
{
    return load_buffer(buffer.data(), buffer.size(), options);
}
program load_buffer(const char* buffer, std::size_t size, const file_options& options)
{
    program p;
    if(options.format == "msgpack")
    {
        p.from_value(from_msgpack(buffer, size));
    }
    else if(options.format == "json")
    {
        p.from_value(from_json_string(buffer, size));
    }
    else
    {
        MIGRAPHX_THROW("Unknown format: " + options.format);
    }
    return p;
}

void save(const program& p, const std::string& filename, const file_options& options)
{
    write_buffer(filename, save_buffer(p, options));
}
std::vector<char> save_buffer(const program& p, const file_options& options)
{
    value v = p.to_value();
    std::vector<char> buffer;
    if(options.format == "msgpack")
    {
        buffer = to_msgpack(v);
    }
    else if(options.format == "json")
    {
        std::string s = to_json_string(v);
        buffer        = std::vector<char>(s.begin(), s.end());
    }
    else
    {
        MIGRAPHX_THROW("Unknown format: " + options.format);
    }
    return buffer;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
