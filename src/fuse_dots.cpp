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
#include <tuple>
#include <string>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static bool is_dot_like(const instruction_ref& ins)
{
    const std::string& n = ins->name();
    return (n == "dot" or n == "quant_dot");
}

// Extract (M,K) from lhs shape using the last two dims.
// Returns false if lhs has rank < 2.
static bool get_mk(const shape& lhs, std::size_t& M, std::size_t& K)
{
    if(lhs.lens().size() < 2) return false;
    const auto& l = lhs.lens();
    M = l[l.size() - 2];
    K = l[l.size() - 1];
    return true;
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

    // For each layer, group dots by (dtype, output shape, M, K) from lhs
    std::cout << "Dot layers: " << dot_layers.size() << std::endl;
    for(const auto& kv : dot_layers)
    {
        int lvl = kv.first;
        const auto& nodes = kv.second;
        std::cout << "  Level " << lvl << ": " << nodes.size() << " dot op(s)" << std::endl;

        // Key: (dtype, out_lens, M, K)
        using key_t = std::tuple<shape::type_t, std::vector<std::size_t>, std::size_t, std::size_t>;
        std::map<key_t, std::vector<instruction_ref>> groups;

        for(auto ins : nodes)
        {
            const auto& out_s = ins->get_shape();
            const auto& lhs_s = ins->inputs().at(0)->get_shape();

            std::size_t M = 0, K = 0;
            if(!get_mk(lhs_s, M, K))
                continue; // skip degenerate cases

            key_t k = std::make_tuple(out_s.type(), out_s.lens(), M, K);
            groups[k].push_back(ins);
        }

        // Print the groups
        for(const auto& gkv : groups)
        {
            const auto& k = gkv.first;
            const auto& vec = gkv.second;
            const auto dtype = std::get<0>(k);
            const auto& out_lens = std::get<1>(k);
            const auto M = std::get<2>(k);
            const auto K = std::get<3>(k);

            // Pretty-print output lens
            std::cout << "    Group [dtype=" << dtype << ", out=[";
            for(std::size_t i = 0; i < out_lens.size(); ++i)
            {
                std::cout << out_lens[i] << (i + 1 < out_lens.size() ? "," : "");
            }
            std::cout << "], M=" << M << ", K=" << K << "]: size=" << vec.size() << std::endl;

            for(auto ins : vec)
            {
                // Print producer lhs and the dot itself for context
                m.debug_print(ins->inputs().front());
                m.debug_print(ins);
                std::cout << std::endl;
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
