#ifndef MIGRAPHX_GUARD_486529_MIGRAPHX_IR_PARSER_HPP
#define MIGRAPHX_GUARD_486529_MIGRAPHX_IR_PARSER_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

program parse_text_ir(const std::string& ir_text);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_486529_MIGRAPHX_IR_PARSER_HPP
