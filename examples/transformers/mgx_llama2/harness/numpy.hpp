#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <regex>
#include <vector>

#include "common.hpp"
#include "logging.hpp"

namespace mlinfer
{
    namespace npy
    {
        class NpyFile
        {
        private:
            std::string m_Path;
            std::ifstream m_FStream;
            size_t m_HeaderSize;
            std::string m_Header;
            size_t m_TensorSize;
            size_t m_ElementSize;
            std::vector<size_t> m_TensorDims;

        public:
            explicit NpyFile(const std::string &path)
                : m_Path(path), m_FStream(m_Path)
            {
                LOG_INFO("Npy file from " << path);
                // magic and fixed header
                char b[256];
                m_FStream.read(b, 10);
                CHECK(m_FStream, "Unable to parse: " << m_Path);

                // check magic
                CHECK(static_cast<unsigned char>(b[0]) == 0x93 && b[1] == 'N' && b[2] == 'U' && b[3] == 'M' && b[4] == 'P' && b[5] == 'Y', "Bad magic: " << m_Path);

                // get header
                auto major = static_cast<uint8_t>(b[6]);
                // auto minor = static_cast<uint8_t>(b[7]);
                CHECK(major == 1, "Only npy version 1 is supported: " << m_Path);
                m_HeaderSize = static_cast<uint16_t>(b[8]);
                m_Header.resize(m_HeaderSize);
                m_FStream.read(static_cast<char *>(m_Header.data()), m_HeaderSize);

                // get file size
                auto cur = m_FStream.tellg();
                m_FStream.seekg(0, std::ios::end);
                auto size = m_FStream.tellg();
                m_TensorSize = size - cur;

                // parse header
                std::regex re(R"re(\{'descr': '[<|][fi]([\d])', 'fortran_order': False, 'shape': \(([\d, ]*)\), \} +\n)re");
                std::smatch matches;
                CHECK(std::regex_match(m_Header, matches, re), "Cannot parse numpy header: " << m_Path);
                CHECK(matches.size() == 3, "Cannot parse numpy header: " << m_Path);
                m_ElementSize = std::stoi(matches[1]);
                std::vector<std::string> dims = splitString(matches[2], ", ");
                m_TensorDims.resize(dims.size());
                std::transform(
                    dims.begin(), dims.end(), m_TensorDims.begin(), [](const std::string &s)
                    { return std::stoi(s); });

                // check header sanity
                size_t tensorSize = std::accumulate(m_TensorDims.begin(), m_TensorDims.end(), m_ElementSize, std::multiplies<size_t>());
                CHECK(tensorSize == m_TensorSize, "Header description does not match file size: " << m_Path);
                LOG_DEBUG("  Input num=" << m_TensorDims[0] << " | Sample size=" << (tensorSize / m_TensorDims[0]) << " | Full size=" << m_TensorSize);
            }
            ~NpyFile()
            {
                m_FStream.close();
            };
            std::string GetPath() const
            {
                return m_Path;
            }
            std::vector<size_t> GetDims() const
            {
                return m_TensorDims;
            }
            size_t GetTensorSize() const
            {
                return m_TensorSize;
            }
            // load the entire tensor
            void LoadAll(void *dst)
            {
                m_FStream.seekg(10 + m_HeaderSize, std::ios::beg);
                m_FStream.read(static_cast<char *>(dst), m_TensorSize);
                CHECK(m_FStream, "Unable to parse: " << m_Path);
                CHECK(m_FStream.peek() == EOF, "Did not consume full file: " << m_Path);
            }

            // load only selected indices from the Tensor, assuming that the first dim is batch dim.
            void LoadSamples(void *dst, const std::vector<size_t> &indices)
            {
                size_t sampleSize = std::accumulate(m_TensorDims.begin() + 1, m_TensorDims.end(), m_ElementSize, std::multiplies<size_t>());
                for (size_t i = 0; i < indices.size(); i++)
                {
                    m_FStream.seekg(10 + m_HeaderSize + indices[i] * sampleSize, std::ios::beg);
                    m_FStream.read(static_cast<char *>(dst) + i * sampleSize, sampleSize);
                }
            }
        };
    } // namespace npy
} // namespace mlinfer

