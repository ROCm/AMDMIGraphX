#include <migraphx/base64.hpp>
#include <vector>
#include <array>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

std::array<char, 64> constexpr B64chars{
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

/// base64 encoder snippet altered from https://stackoverflow.com/a/37109258
const std::string b64_encode(const void* data, const size_t& len)
{
    std::vector<char> res_vec((len + 2) / 3 * 4, '=');
    const unsigned char* p = static_cast<const unsigned char*>(data);
    size_t j = 0, pad = len % 3;
    const size_t last = len - pad;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
    for(size_t i = 0; i < last; i += 3)
    {
        unsigned n =
            static_cast<unsigned>(p[i]) << 16 | static_cast<unsigned>(p[i + 1]) << 8 | p[i + 2];
        res_vec.at(j++) = B64chars.at(n >> 18);
        res_vec.at(j++) = B64chars.at(n >> 12 & 0x3F);
        res_vec.at(j++) = B64chars.at(n >> 6 & 0x3F);
        res_vec.at(j++) = B64chars.at(n & 0x3F);
    }
    if(pad) /// Set padding
    {
        unsigned n      = --pad ? static_cast<unsigned>(p[last]) << 8 | p[last + 1] : p[last];
        res_vec.at(j++) = B64chars.at(pad ? n >> 10 & 0x3F : n >> 2);
        res_vec.at(j++) = B64chars.at(pad ? n >> 4 & 0x03F : n << 4 & 0x3F);
        res_vec.at(j++) = pad ? B64chars.at(n << 2 & 0x3F) : '=';
    }
#pragma clang diagnostic pop
    return std::string(res_vec.begin(), res_vec.end());
}

} // namespace

std::string b64_encode(const std::string& str) { return b64_encode(str.c_str(), str.size()); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
