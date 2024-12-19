#include <migraphx/base64.hpp>
#include <vector>
#include <array>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
typedef unsigned char byte;

std::array<char, 64> constexpr B64chars{
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

/// base64 encoder snippet altered from https://stackoverflow.com/a/37109258
std::string b64_encode(const std::vector<byte> buf)
{
    std::size_t len = buf.size();
    std::vector<byte> res_vec((len + 2) / 3 * 4, '=');
    std::size_t j        = 0;
    std::size_t pad_cond = len % 3;
    const size_t last    = len - pad_cond;

    for(size_t i = 0; i < last; i += 3)
    {
        std::size_t n = static_cast<std::size_t>(buf.at(i)) << 16u |
                        static_cast<std::size_t>(buf.at(i + 1)) << 8u |
                        static_cast<std::size_t>(buf.at(i + 2));
        res_vec.at(j++) = B64chars.at(n >> 18u);
        res_vec.at(j++) = B64chars.at(n >> 12u & 0x3F);
        res_vec.at(j++) = B64chars.at(n >> 6u & 0x3F);
        res_vec.at(j++) = B64chars.at(n & 0x3F);
    }
    if(pad_cond) /// Set padding
    {
        std::size_t n   = --pad_cond ? static_cast<std::size_t>(buf.at(last)) << 8u |
                                         static_cast<std::size_t>(buf.at(last + 1))
                                     : static_cast<std::size_t>(buf.at(last));
        res_vec.at(j++) = B64chars.at(pad_cond ? n >> 10u & 0x3F : n >> 2u);
        res_vec.at(j++) = B64chars.at(pad_cond ? n >> 4u & 0x03F : n << 4u & 0x3F);
        res_vec.at(j++) = pad_cond ? B64chars.at(n << 2u & 0x3F) : '=';
    }
    return std::string(res_vec.begin(), res_vec.end());
}

} // namespace

std::string b64_encode(const std::string& str)
{
    return b64_encode(std::vector<byte>(str.begin(), str.end()));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
