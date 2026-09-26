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
#include <migraphx/system.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif
#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef __linux__
// The list of logical CPUs sharing a core with cpu, which names the core. A CPU whose topology
// can't be read gets a name of its own, so it counts as a core.
static std::string core_name(std::ptrdiff_t cpu)
{
    auto path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    std::ifstream siblings{path + "/topology/thread_siblings_list"};
    std::string name;
    if(std::getline(siblings, name))
        return name;
    return path;
}
#endif

std::size_t physical_cpu_cores()
{
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if(sched_getaffinity(0, sizeof(allowed), &allowed) == 0)
    {
        std::set<std::string> cores;
        auto cpus = range(CPU_SETSIZE);
        transform_if(
            cpus.begin(),
            cpus.end(),
            std::inserter(cores, cores.end()),
            [&](auto cpu) { return CPU_ISSET(cpu, &allowed) != 0; },
            [](auto cpu) { return core_name(cpu); });
        if(not cores.empty())
            return cores.size();
    }
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}

void trim_heap()
{
#ifdef __GLIBC__
    malloc_trim(0);
#endif
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
