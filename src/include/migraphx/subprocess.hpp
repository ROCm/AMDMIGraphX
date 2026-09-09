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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SUBPROCESS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SUBPROCESS_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct subprocess_result
{
    int exit_code = 0;
    std::vector<char> stdout_data{};
};

/// Spawn `exe` with `argv` (not including the executable), feed `stdin_data` to the child while
/// concurrently draining its stdout, and wait for it to exit. stderr is inherited, so the child's
/// diagnostics go wherever ours go. Throws if the child cannot be spawned; a non-zero exit status
/// is returned in `exit_code`, not thrown. Only compare `exit_code` against zero -- a child killed
/// by a signal or an SEH exception reports -1.
MIGRAPHX_EXPORT subprocess_result execute_subprocess(const fs::path& exe,
                                                     const std::vector<std::string>& argv,
                                                     const std::vector<char>& stdin_data);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SUBPROCESS_HPP
