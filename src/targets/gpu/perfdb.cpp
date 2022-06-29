#include <migraphx/gpu/perfdb.hpp>
#include <migraphx/value.hpp>
#include <migraphx/sqlite.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/permutation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

std::string get_layout(const shape& s, std::string labels)
{
    auto result = labels;
    auto p      = find_permutation(s);
    std::transform(p.begin(), p.end(), result.begin(), [&](auto i) { return labels[i]; });
    return result;
}

std::string get_type(const shape& s)
{
    static const std::unordered_map<shape::type_t, std::string> m = {
        {shape::float_type, "FP32"},
        {shape::half_type, "FP16"},
        {shape::double_type, "FP64"},
        {shape::int8_type, "INT8"},
        {shape::int32_type, "INT32"},
    };
    auto it = m.find(s.type());
    if(it == m.end())
        return "UNKNOWN";
    return it->second;
}

std::string generate_miopen_config(const problem_params& pp)
{
    value v       = pp.op.to_value();
    auto input    = pp.inputs[0].lens();
    auto weights  = pp.inputs[1].lens();
    auto output   = pp.output.lens();
    auto padding  = v["padding"].to_vector<std::size_t>();
    auto stride   = v["stride"].to_vector<std::size_t>();
    auto dilation = v["dilation"].to_vector<std::size_t>();
    if(padding.size() != stride.size())
        padding.erase(padding.begin() + padding.size() / 2, padding.end());

    return to_string_range({to_string(input[1]),
                            to_string_range(input.begin() + 2, input.end(), "x"),
                            to_string_range(weights.begin() + 2, weights.end(), "x"),
                            to_string(weights[0]),
                            to_string_range(output.begin() + 2, output.end(), "x"),
                            to_string(input[0]),
                            to_string_range(padding.begin() + 2, padding.end(), "x"),
                            to_string_range(stride.begin() + 2, stride.end(), "x"),
                            to_string_range(dilation.begin() + 2, dilation.end(), "x"),
                            std::string{"0"},
                            get_layout(pp.inputs[0], "NCHW"),
                            get_layout(pp.inputs[1], "NCHW"),
                            get_layout(pp.output, "NCHW"),
                            get_type(pp.inputs[0]),
                            std::string{"F"}},
                           "-");
}

} // namespace

std::string get_mlir_perf_for_conv(const problem_params& pp)
{
    const auto dbpath = fs::path{"opt"} / "rocm" / "share" / "miopen" / "db" / "miopen.db";
    auto db           = sqlite::read(dbpath);
    std::string query = "select * from perf_db where config=${config}";
    auto results = db.execute(interpolate_string(query, {{"config", generate_miopen_config(pp)}}));
    if(results.empty())
        return "";
    return results.front().at("params");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
