#include <migraphx/base64.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// base64 encoder snippet altered from https://stackoverflow.com/a/37109258
std::string b64_encode(const std::string& input)
{
    static const std::string B64chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const std::size_t len = input.size();
    std::vector<char> res_vec((len + 2) / 3 * 4, '=');
    std::size_t j     = 0;
    std::size_t pad   = len % 3;
    const size_t last = len - pad;

    for(size_t i = 0; i < last; i += 3)
    {
        int n = static_cast<int>(input.at(i)) << 16 | static_cast<int>(input.at(i + 1)) << 8 |
                input.at(i + 2);
        res_vec.at(j++) = B64chars.at(n >> 18);
        res_vec[j++]    = B64chars.at(n >> 12 & 0x3F);
        res_vec[j++]    = B64chars.at(n >> 6 & 0x3F);
        res_vec[j++]    = B64chars.at(n & 0x3F);
    }
    if(pad) /// Set padding
    {
        int n = --pad ? static_cast<int>(input.at(last)) << 8 | input.at(last + 1) : input.at(last);
        res_vec[j++] = B64chars.at(pad ? n >> 10 & 0x3F : n >> 2);
        res_vec[j++] = B64chars.at(pad ? n >> 4 & 0x03F : n << 4 & 0x3F);
        res_vec[j++] = pad ? B64chars.at(n << 2 & 0x3F) : '=';
    }
    return std::string(res_vec.begin(), res_vec.end());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
