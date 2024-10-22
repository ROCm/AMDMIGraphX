#pragma once

#include <cstring>
#include <numaif.h>
#include <numeric>
#include <pthread.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <fstream>

#include "common.hpp"

namespace mlinfer
{
    // NUMA config. Each NUMA node contains a pair of GPU indices and CPU indices.
    using NumaConfig = std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>;

    // The NUMA node idx for each GPU.
    using GpuToNumaMap = std::vector<size_t>;

    struct NumaSettings
    {
        NumaConfig numa_config;
        GpuToNumaMap gpu_to_numa_map;
    };

    struct Numa final
    {
        NumaSettings numa_settings;

        explicit Numa(const NumaSettings &numa_settings) : numa_settings{numa_settings} {}

        inline bool UseNuma() const
        {
            return not numa_settings.numa_config.empty();
        }

        inline size_t GetNumaCount() const
        {
            return numa_settings.numa_config.size();
        };

        inline int GetNumaIdx(const int deviceId) const
        {
            return UseNuma() ? numa_settings.gpu_to_numa_map.at(deviceId) : 0;
        }

        inline std::vector<size_t> GetClosestCpus(const int deviceId) const
        {
            assertm(UseNuma(), "GetClosestCpus only available for NUMA");
            return numa_settings.numa_config.at(GetNumaIdx(deviceId)).second;
        }
    };

    // Restrict mem allocation to specific NUMA node.
    inline void
    bindNumaMemPolicy(const int32_t numaIdx, const int32_t nbNumas)
    {
        unsigned long nodeMask = 1UL << numaIdx;
        long ret = set_mempolicy(MPOL_BIND, &nodeMask, nbNumas + 1);
        CHECK(ret >= 0, std::strerror(errno));
    }

    // Reset mem allocation setting.
    inline void resetNumaMemPolicy()
    {
        long ret = set_mempolicy(MPOL_DEFAULT, nullptr, 0);
        CHECK(ret >= 0, std::strerror(errno));
    }

    // Limit a thread to be on specific cpus.
    inline void bindThreadToCpus(std::thread &th, const std::vector<size_t> &cpus, const bool ignore_esrch = false)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int cpu : cpus)
        {
            CPU_SET(cpu, &cpuset);
        }
        int ret = pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
        bool noerr = ignore_esrch ? ret == 0 || ret == ESRCH : ret == 0;
        CHECK(noerr, std::strerror(ret));
    }

    // Helper to converts the range string (like "0,2-5,13-17") to a vector of ints.
    inline std::vector<size_t> parseRange(const std::string &s)
    {
        std::vector<size_t> results;
        auto ranges = splitString(s, ",");
        for (const auto &range : ranges)
        {
            auto startEnd = splitString(range, "-");
            CHECK((startEnd.size() <= 2), "Invalid numa_config setting. Expects zero or one '-'.");
            if (startEnd.size() == 1)
            {
                results.push_back(std::stoi(startEnd[0]));
            }
            else
            {
                size_t start = std::stoi(startEnd[0]);
                size_t last = std::stoi(startEnd[1]);
                for (size_t i = start; i <= last; ++i)
                {
                    results.push_back(i);
                }
            }
        }
        return results;
    }

    // Example of the format: "0,2:0-63&1,3:64-127" for 4 GPUs, 128 CPU, 2 NUMA node system.
    inline NumaConfig parseNumaConfig(const std::string &numa_file)
    {
        std::string numa_str;
        std::ifstream file(numa_file.c_str());
        if (file.is_open())
        {
            getline(file, numa_str);
            file.close();
        }

        NumaConfig config;
        if (!numa_str.empty())
        {
            auto nodes = splitString(numa_str, "&");
            for (const auto &node : nodes)
            {
                auto pair = splitString(node, ":");
                CHECK((pair.size() == 2), "Invalid numa_config setting. Expects one ':'.");
                auto gpus = parseRange(pair[0]);
                auto cpus = parseRange(pair[1]);
                config.emplace_back(std::make_pair(gpus, cpus));
            }
        }
        return config;
    }

    // Convert NumaConfig to GpuToNumaMap for easier look-up.
    inline GpuToNumaMap getGpuToNumaMap(const NumaConfig &config)
    {
        std::vector<size_t> map;
        for (size_t numaIdx = 0; numaIdx < config.size(); numaIdx++)
        {
            for (const auto gpuIdx : config[numaIdx].first)
            {
                if (gpuIdx >= map.size())
                {
                    map.resize(gpuIdx + 1);
                }
                map[gpuIdx] = numaIdx;
            }
        }
        return map;
    }
} // namespace mlinfer

