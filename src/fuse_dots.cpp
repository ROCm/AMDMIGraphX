// fuse_dots.cpp
#include <migraphx/fuse_dots.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/iterator_for.hpp>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {


static bool is_dot_like(const instruction_ref& ins)
{
    const std::string& n = ins->name();
    return (n == "dot" or n == "quant_dot");
}

void fuse_dots::apply(module& m) const
{
    // Build indegree map and initialize levels
    std::unordered_map<instruction_ref, std::size_t> indeg;
    indeg.reserve(std::distance(m.begin(), m.end()));
    std::unordered_map<instruction_ref, int> level;
    level.reserve(indeg.size());

    for(auto ins : iterator_for(m))
    {
        indeg[ins] = ins->inputs().size();
    }

    // Kahn’s algorithm queue initialization
    std::queue<instruction_ref> q;
    for(auto ins : iterator_for(m))
    {
        if(indeg.at(ins) == 0)
        {
            level[ins] = 0;
            q.push(ins);
        }
    }

    // Forward BFS/toposort computing levels
    while(!q.empty())
    {
        auto u = q.front();
        q.pop();
        int ulevel = level[u];

        for(auto v : u->outputs())
        {
            // Level is 1 + max(parent levels)
            auto it = level.find(v);
            int cand = ulevel + 1;
            if(it == level.end())
                level[v] = cand;
            else
                it->second = std::max(it->second, cand);

            // Decrement indegree and enqueue when ready
            auto dit = indeg.find(v);
            if(dit != indeg.end() && dit->second > 0)
            {
                dit->second -= 1;
                if(dit->second == 0)
                    q.push(v);
            }
        }
    }

    // Group dot-like ops by layer
    std::map<int, std::vector<instruction_ref>> dot_layers;
    for(auto ins : iterator_for(m))
    {
        if(is_dot_like(ins))
        {
            int lvl = 0;
            auto it = level.find(ins);
            if(it != level.end())
                lvl = it->second;
            dot_layers[lvl].push_back(ins);
        }
    }

    // Example: print summary (replace with your preferred reporting)
    std::cout << "Dot layers: " << dot_layers.size() << std::endl;
    for(const auto& kv : dot_layers)
    {
        std::cout << "  Level " << kv.first << ": " << kv.second.size() << " dot op(s)" << std::endl;
        for(auto ins : kv.second)
        {
            m.debug_print(ins->inputs().front());
            m.debug_print(ins);
            std::cout << std::endl;
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
