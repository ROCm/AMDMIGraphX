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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_EXTERNAL_DATA_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_EXTERNAL_DATA_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/functional.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct mapped_file;

/// Byte alignment of every literal written to an external weights file
constexpr std::size_t external_data_alignment = 64;
/// Default size at or above which a literal is stored externally
constexpr std::size_t default_external_data_threshold = 128 * 1024;
/// Appended to the program file name to form the default weights file base name
constexpr const char* external_data_extension = ".data";

/// Where the bytes of a literal are stored outside the program file
struct external_data
{
    /// Relative to the program file's directory, or absolute
    std::string file_path;
    std::size_t offset = 0;
    std::size_t size   = 0;
    /// md5 hex digest of the bytes, an integrity check rather than tamper protection; empty until
    /// computed
    std::string checksum;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.file_path, "file_path"),
                    f(self.offset, "offset"),
                    f(self.size, "size"),
                    f(self.checksum, "checksum"));
    }
};

/// Where a literal's data was read from, and the mapping that keeps it alive; shared by the
/// literal's copies
struct external_data_source
{
    external_data ref;
    std::shared_ptr<mapped_file> file;
};

struct external_data_save_options
{
    /// Data of at least this many bytes is written to a new generation of weights_base; nullopt
    /// writes none
    std::optional<std::size_t> threshold = std::nullopt;
    /// Data that already has an external reference keeps it instead of being written again
    bool reuse_external_data = false;
    /// Must be a nonzero power of two
    std::size_t alignment = external_data_alignment;
    /// Each writer stores new data in its own file, see weights_generation_path
    fs::path weights_base;
    /// References to files under it are written relative to it; empty writes absolute paths
    fs::path base_dir;
};

struct external_data_load_options
{
    /// Relative references resolve against it, and every reference must lie within it or one of
    /// allowed_dirs; empty uses the current directory
    fs::path base_dir;
    /// Further directories references may point into
    std::vector<fs::path> allowed_dirs;
    bool verify_checksum = false;
};

struct external_data_writer_impl;

/**
 * @brief Stores literal and argument data externally for a serialize_context that names it
 * @details New data goes to a temporary file that commit() renames to a generation of the
 * weights base that no other writer uses, so a published file is never overwritten; without a
 * commit the temporary is removed.
 */
struct MIGRAPHX_EXPORT external_data_writer
{
    explicit external_data_writer(external_data_save_options options);
    ~external_data_writer();

    external_data_writer(const external_data_writer&)            = delete;
    external_data_writer& operator=(const external_data_writer&) = delete;

    /// The reference to store instead of the n bytes at data, or nullopt to keep them inline.
    /// original is the reference the data was read from, if any.
    std::optional<external_data>
    write(const char* data, std::size_t n, const external_data* original);

    void commit();

    /// Remove the generation commit() published, for when the program file referencing it
    /// could not be published
    void discard();

    /// Whether write() has returned a reference, new or reused
    bool has_references() const;

    /// Absolute paths of the files the returned references name
    const std::vector<fs::path>& referenced_files() const;

    private:
    std::unique_ptr<external_data_writer_impl> impl;
};

struct external_data_reader_impl;

/// Resolves the external references of a serialize_context that names it against its options,
/// sharing one mapping per file
struct MIGRAPHX_EXPORT external_data_reader
{
    explicit external_data_reader(external_data_load_options options);
    ~external_data_reader();

    external_data_reader(const external_data_reader&)            = delete;
    external_data_reader& operator=(const external_data_reader&) = delete;

    const external_data_load_options& get_options() const;

    /// The path of the file a reference names, which must exist and, unless the options say
    /// otherwise, lie within the base directory or an allowed directory after resolving symlinks
    fs::path resolve(const std::string& file_path) const;

    /// The mapping of the file at an already resolved path
    std::shared_ptr<mapped_file> map(const fs::path& resolved);

    /// Map the file a reference points at. The bounds are always checked and the checksum when
    /// the options ask for it. The returned reference holds the resolved absolute path.
    std::pair<std::shared_ptr<mapped_file>, external_data> read(const external_data& ref);

    private:
    std::unique_ptr<external_data_reader_impl> impl;
};

/// Lexically resolve '.' and '..' without touching the filesystem
MIGRAPHX_EXPORT fs::path lexically_normalize(const fs::path& p);

/// base_dir / relative, which must stay within base_dir
MIGRAPHX_EXPORT fs::path resolve_external_data_path(const fs::path& base_dir,
                                                    const fs::path& relative);

/// p relative to base_dir when it lies under it, otherwise absolute
MIGRAPHX_EXPORT std::string relative_external_data_path(const fs::path& base_dir,
                                                        const fs::path& p);

/// The generation id of base: <stem>.<id><extension> beside it, e.g. model.mxr.<id>.data
MIGRAPHX_EXPORT fs::path weights_generation_path(const fs::path& base, const std::string& id);

/// Whether p names a generation of base, as weights_generation_path writes them
MIGRAPHX_EXPORT bool is_weights_generation(const fs::path& base, const fs::path& p);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
