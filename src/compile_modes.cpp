/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/compile_modes.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/stringutils.hpp>
#include <cstdlib>
#include <algorithm>
#include <array>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {


compile_modes convert_to_compile_mode(const uint8_t mode)
{
    auto clamped = static_cast<uint8_t>(std::clamp<int>(mode, 0, 100));
    if(clamped != mode)
        log::warn() << "Compile mode value " << static_cast<int>(mode)
                     << " out of range [0, 100], clamping to " << static_cast<int>(clamped);

    static const std::array<compile_modes, 3> modes = {
        compile_modes::EAGER, compile_modes::BALANCED, compile_modes::MAX};
        
    auto it = std::find_if(modes.begin(), modes.end(), [&](compile_modes m) {
        return static_cast<uint8_t>(m) == clamped;
    });
    if(it != modes.end())
        return *it;

    log::warn() << "Compile mode value " << static_cast<int>(clamped)
                << " does not match a known mode, using closest match";
    return *std::min_element(modes.begin(), modes.end(), by(std::less<>{}, [&](compile_modes m) {
        return std::abs(static_cast<int>(clamped) - static_cast<int>(m));
    }));
}

compile_modes convert_to_compile_mode(const std::string& mode)
{
    auto lower = to_lower(mode);
    if(lower == "eager")
        return compile_modes::EAGER;
    if(lower == "balanced")
        return compile_modes::BALANCED;
    if(lower == "max")
        return compile_modes::MAX;
    try
    {
        int val = std::stoi(mode);
        if(val < 0 or val > 100)
            log::warn() << "Compile mode value " << val
                        << " out of range [0, 100], clamping to " << std::clamp(val, 0, 100);
        return convert_to_compile_mode(static_cast<uint8_t>(std::clamp(val, 0, 100)));
    }
    catch(const std::invalid_argument&)
    {
        MIGRAPHX_THROW("Invalid compile mode: " + mode +
                       ". Expected eager, balanced, max, or an integer 0-100");
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
