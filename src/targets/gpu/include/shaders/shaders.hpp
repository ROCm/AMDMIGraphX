/*
 * Minimal C++17-compatible shim for AMDMLSS StaticShaderType.
 *
 * The real shaders/shaders.hpp (from AMDMLSS) requires C++20 (std::span, std::ranges).
 * MIGraphX builds with C++17, so this header provides just enough to compile the
 * shadersBin*.hpp data headers which only need StaticShaderType and
 * ShaderTypesFlags::{UNKNOWN,WMMA}.
 */
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace mlss {
enum class ShaderTypesFlags : std::uint32_t { UNKNOWN = 0, WMMA = 1 };
} // namespace mlss

template <std::size_t M>
struct StaticShaderType
{
    std::array<std::uint8_t, M> m_binary;
    std::string_view            m_kernelName;
    std::string_view            m_compilerVersion;
    std::uint32_t               m_codeObjectVersion;
    bool                        m_isRelocatable;
    mlss::ShaderTypesFlags      m_shaderType;
};
