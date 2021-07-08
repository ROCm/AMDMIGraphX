#include <migraphx/compile_src.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/errors.hpp>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<char> src_compiler::compile(const std::vector<src_file>& srcs) const
{
    assert(not srcs.empty());
    tmp_dir td{"compile"};
    auto params = flags;

    params += " -I.";

    auto out = output;

    for(const auto& src : srcs)
    {
        fs::path full_path   = td.path / src.path;
        fs::path parent_path = full_path.parent_path();
        fs::create_directories(parent_path);
        write_buffer(full_path.string(), src.content.first, src.len());
        if(src.path.extension().string() == ".cpp")
        {
            params += " " + src.path.filename().string();
            if(out.empty())
                out = src.path.stem().string() + ".o";
        }
    }

    params += " -o " + out;

    td.execute(compiler, params);

    auto out_path = td.path / out;
    if(not fs::exists(out_path))
        MIGRAPHX_THROW("Output file missing: " + out);

    if(process)
        out_path = process(out_path);

    return read_buffer(out_path.string());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
