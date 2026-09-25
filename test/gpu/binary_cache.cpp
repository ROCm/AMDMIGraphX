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
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/json.hpp>
#include <migraphx/md5.hpp>
#include <migraphx/sqlite.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/binary_cache.hpp>
#include <migraphx/gpu/binary_cache_backend.hpp>
#include <migraphx/gpu/file_binary_cache.hpp>
#include <migraphx/gpu/sqlite_binary_cache.hpp>
#include <migraphx/gpu/compiled_code.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>

static migraphx::program pointwise_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    auto x    = mm->add_parameter("x", s);
    auto y    = mm->add_parameter("y", s);
    auto add  = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto relu = mm->add_instruction(migraphx::make_op("relu"), add);
    auto mul  = mm->add_instruction(migraphx::make_op("mul"), relu, x);
    mm->add_return({mul});
    return p;
}

static migraphx::compile_options cache_options(const std::string& path, bool verify = false)
{
    migraphx::compile_options options;
    migraphx::set_backend_options(options,
                                  {{"binary_cache", path}, {"binary_cache_verify", verify}});
    return options;
}

static migraphx::gpu::compiled_code make_code()
{
    migraphx::gpu::compiled_code code;
    auto* fm = code.fragment.get_main_module();
    auto x   = fm->add_parameter(migraphx::gpu::compiled_code::input_name(0),
                                 {migraphx::shape::float_type, {2}});
    fm->add_return({fm->add_instruction(migraphx::make_op("abs"), x)});
    code.fill_map["float_type{2}"] = 3.0;
    return code;
}

static migraphx::gpu::binary_cache::entry make_entry(const std::string& key)
{
    migraphx::gpu::binary_cache::entry e;
    e.key      = key;
    e.op_name  = "pointwise";
    e.problem  = migraphx::value{{"shape", "float_type{4, 8}"}};
    e.solution = migraphx::value{{"algo", "block"}};
    e.code     = make_code();
    return e;
}

// The storage backend is chosen by the extension of the cache path.
static std::string dir_path(const migraphx::tmp_dir& td) { return td.path.string(); }
static std::string db_path(const migraphx::tmp_dir& td) { return (td.path / "cache.db").string(); }

/// The entry files a directory-backed cache has written.
static std::vector<migraphx::fs::path> entry_files(const migraphx::fs::path& dir)
{
    std::vector<migraphx::fs::path> result;
    migraphx::transform_if(
        migraphx::fs::recursive_directory_iterator{dir},
        migraphx::fs::recursive_directory_iterator{},
        std::back_inserter(result),
        [](const auto& file) { return file.path().extension() == ".mxr"; },
        [](const auto& file) { return file.path(); });
    return result;
}

/// Rows in one table of a cache database. The count is aliased because sqlite::execute keys its
/// rows by column name, and an unaliased count(*) would be keyed by the text of the expression.
static std::size_t row_count(const std::string& path, const std::string& table)
{
    auto rows = migraphx::sqlite::read(path).execute("SELECT count(*) AS n FROM " + table + ";");
    if(rows.empty())
        return 0;
    return std::stoul(rows.front().at("n"));
}

using stored_entries = std::map<std::string, std::vector<char>>;

/// Every entry a cache directory holds, keyed by key hash, which names each file.
static stored_entries dir_entries(const migraphx::fs::path& dir)
{
    stored_entries result;
    auto files = entry_files(dir);
    std::transform(files.begin(), files.end(), std::inserter(result, result.end()), [](auto f) {
        return std::make_pair(f.stem().string(), migraphx::read_buffer(f));
    });
    return result;
}

/// Every entry a cache database holds, keyed by key hash.
static stored_entries db_entries(const std::string& path)
{
    stored_entries result;
    auto select = migraphx::sqlite::read(path).prepare("SELECT key_hash, entry FROM cache_v1;");
    auto rows   = select();
    std::transform(rows.begin(), rows.end(), std::inserter(result, result.end()), [](auto row) {
        const auto& blob = row.at("entry").get_binary();
        return std::make_pair(row.at("key_hash").get_string(),
                              std::vector<char>(blob.begin(), blob.end()));
    });
    return result;
}

// What a case that must hold for both backends needs to know about each, so the case can be
// written once as a template and registered for both.
struct directory_backend
{
    static std::string path(const migraphx::tmp_dir& td) { return dir_path(td); }
    static std::size_t stored(const std::string& p) { return entry_files(p).size(); }
    /// Overwrite every stored entry with bytes that do not decode.
    static void damage(const std::string& p)
    {
        auto files = entry_files(p);
        std::for_each(files.begin(), files.end(), [](const auto& file) {
            migraphx::write_buffer(file, std::vector<char>(8, 0));
        });
    }
};

struct database_backend
{
    static std::string path(const migraphx::tmp_dir& td) { return db_path(td); }
    static std::size_t stored(const std::string& p) { return row_count(p, "cache_v1"); }
    /// Overwrite every stored entry with bytes that do not decode.
    static void damage(const std::string& p)
    {
        migraphx::sqlite::write(p).execute("UPDATE cache_v1 SET entry = zeroblob(8);");
    }
};

/// One of each backend over fresh storage in td: a directory at td/files and a database at
/// db_path(td). Driven directly, these skip binary_cache and its version and device strings.
static std::vector<migraphx::gpu::binary_cache_backend> both_backends(const migraphx::tmp_dir& td)
{
    std::vector<migraphx::gpu::binary_cache_backend> result;
    result.emplace_back(migraphx::gpu::file_binary_cache{td.path / "files"});
    auto db = migraphx::gpu::sqlite_binary_cache::open(db_path(td));
    EXPECT(db.has_value());
    if(db.has_value())
        result.emplace_back(std::move(*db));
    return result;
}

TEST_CASE(lookup_records_a_miss)
{
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{.path = ""}};

    EXPECT(not cache.get(ctx, "absent").has_value());
    EXPECT(cache.get_stats().misses == 1);
    EXPECT(cache.get_stats().hits == 0);
    EXPECT(cache.get_stats().reused == 0);
}

// An entry served out of memory is one an earlier compile in this process already paid for.
TEST_CASE(memory_lookup_records_reuse)
{
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{.path = ""}};

    cache.insert(ctx, make_entry("a-key"));
    EXPECT(cache.get_stats().compiled == 1);

    auto found = cache.get(ctx, "a-key");
    EXPECT(found.has_value());
    EXPECT(cache.get_stats().reused == 1);
    EXPECT(cache.get_stats().misses == 0);
}

// The cases below are written once against a Backend and registered for each. The directory
// registrations use real temporary paths so that on Windows they exercise the full depth of an
// entry path against MAX_PATH.

// A second cache shares nothing in memory, so anything it finds came out of storage.
template <class Backend>
static void disk_lookup_records_a_hit()
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache_settings settings{Backend::path(td), false};

    migraphx::gpu::binary_cache writer{settings};
    writer.insert(ctx, make_entry("shared-key"));

    migraphx::gpu::binary_cache reader{settings};

    auto found = reader.get(ctx, "shared-key");
    EXPECT(found.has_value());
    EXPECT(reader.get_stats().hits == 1);
    EXPECT(reader.get_stats().misses == 0);
    EXPECT(*found->fragment.get_main_module() == *make_code().fragment.get_main_module());
}
TEST_CASE_REGISTER(disk_lookup_records_a_hit<directory_backend>);
TEST_CASE_REGISTER(disk_lookup_records_a_hit<database_backend>);

// A damaged entry must cost a recompile and nothing more.
template <class Backend>
static void corrupt_entry_is_ignored()
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = Backend::path(td);
    migraphx::gpu::binary_cache_settings settings{path, false};

    migraphx::gpu::binary_cache writer{settings};
    writer.insert(ctx, make_entry("damaged"));
    EXPECT(Backend::stored(path) == 1);
    Backend::damage(path);

    migraphx::gpu::binary_cache reader{settings};

    EXPECT(not reader.get(ctx, "damaged").has_value());
    EXPECT(reader.get_stats().misses == 1);
}
TEST_CASE_REGISTER(corrupt_entry_is_ignored<directory_backend>);
TEST_CASE_REGISTER(corrupt_entry_is_ignored<database_backend>);

// Without a directory nothing reaches disk, though results are still shared in memory.
TEST_CASE(no_directory_writes_nothing)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{.path = ""}};

    cache.insert(ctx, make_entry("in-memory-only"));
    EXPECT(cache.get(ctx, "in-memory-only").has_value());
    EXPECT(cache.get_stats().reused == 1);
    EXPECT(migraphx::fs::is_empty(td.path));
}

static migraphx::program two_identical_pointwise()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto pw1 = add_pointwise(p, "main:pointwise0", {x, y}, single_pointwise("add"));
    auto pw2 = add_pointwise(p, "main:pointwise1", {y, x}, single_pointwise("add"));
    mm->add_return({pw1, pw2});
    return p;
}

static std::size_t count_code_objects(const migraphx::module& m)
{
    return std::count_if(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "gpu::code_object"; });
}

// The in-memory cache needs no directory: two identical kernels in one model compile once, and
// a later module compiled with the same context reuses the result without compiling at all.
TEST_CASE(duplicate_kernels_compile_once_without_a_directory)
{
    auto cache = std::make_shared<migraphx::gpu::binary_cache>(
        migraphx::gpu::binary_cache_settings{.path = ""});
    migraphx::gpu::context ctx{0, 1};
    ctx.set_binary_cache(cache);

    auto p1 = two_identical_pointwise();
    migraphx::run_passes(*p1.get_main_module(),
                         {migraphx::gpu::lowering{&ctx, false}, migraphx::gpu::compile_ops{&ctx}});
    EXPECT(count_code_objects(*p1.get_main_module()) == 2);
    EXPECT(cache->get_stats().compiled == 1);

    auto p2 = two_identical_pointwise();
    migraphx::run_passes(*p2.get_main_module(),
                         {migraphx::gpu::lowering{&ctx, false}, migraphx::gpu::compile_ops{&ctx}});
    EXPECT(count_code_objects(*p2.get_main_module()) == 2);
    EXPECT(cache->get_stats().compiled == 1);
    EXPECT(cache->get_stats().reused == 1);
}

// Compiling twice against the same cache has to leave entries behind and keep producing the
// same numbers as the reference, whichever half of the run they came from.
template <class Backend>
static void compiling_twice_populates_the_cache_and_matches_reference()
{
    migraphx::tmp_dir td{"binary-cache"};
    auto path    = Backend::path(td);
    auto options = cache_options(path);

    auto p_ref = pointwise_program();
    p_ref.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    auto x = migraphx::generate_argument(s, 0);
    auto y = migraphx::generate_argument(s, 1);

    auto ref_result = p_ref.eval({{"x", x}, {"y", y}}).back();

    auto warmup = pointwise_program();
    warmup.compile(migraphx::make_target("gpu"), options);

    EXPECT(Backend::stored(path) > 0);

    auto t = migraphx::make_target("gpu");
    auto p = pointwise_program();
    p.compile(t, options);

    migraphx::parameter_map params;
    for(auto&& [name, shape] : p.get_parameter_shapes())
    {
        if(name == "x")
            params[name] = t.copy_to(x);
        else if(name == "y")
            params[name] = t.copy_to(y);
        else
            params[name] = t.allocate(shape);
    }
    auto gpu_result = t.copy_from(p.eval(params).back());

    EXPECT(migraphx::verify::verify_rms_range(ref_result.to_vector<float>(),
                                              gpu_result.to_vector<float>()));
}

TEST_CASE_REGISTER(compiling_twice_populates_the_cache_and_matches_reference<directory_backend>);
TEST_CASE_REGISTER(compiling_twice_populates_the_cache_and_matches_reference<database_backend>);

// With verification on, every reused result is compiled again and compared, so a run that does
// not throw is one where the keys really do capture what the compilers depend on.
template <class Backend>
static void verified_reuse_matches_fresh_compiles()
{
    migraphx::tmp_dir td{"binary-cache"};
    auto options = cache_options(Backend::path(td), /* verify */ true);

    auto warmup = pointwise_program();
    warmup.compile(migraphx::make_target("gpu"), options);

    auto p = pointwise_program();
    p.compile(migraphx::make_target("gpu"), options);
}

TEST_CASE_REGISTER(verified_reuse_matches_fresh_compiles<directory_backend>);
TEST_CASE_REGISTER(verified_reuse_matches_fresh_compiles<database_backend>);

// The extension of the path picks the backend and nothing else does, so the only way to see the
// choice from outside is the artifact it leaves: a database file, or a directory tree.
TEST_CASE(extension_selects_the_backend)
{
    migraphx::gpu::context ctx;
    const auto& version_dir = migraphx::gpu::binary_cache::version_id(true);

    migraphx::tmp_dir dir_td{"binary-cache"};
    migraphx::gpu::binary_cache dir_cache{
        migraphx::gpu::binary_cache_settings{dir_path(dir_td), false}};
    dir_cache.insert(ctx, make_entry("in-a-directory"));
    auto files = entry_files(dir_td.path);
    EXPECT(files.size() == 1);
    EXPECT(migraphx::fs::is_directory(dir_td.path / version_dir));
    EXPECT(std::all_of(files.begin(), files.end(), [&](const auto& f) {
        return f.parent_path().parent_path() == dir_td.path / version_dir;
    }));

    for(const char* name : {"cache.db", "cache.sqlite"})
    {
        migraphx::tmp_dir db_td{"binary-cache"};
        auto path = (db_td.path / name).string();
        migraphx::gpu::binary_cache db_cache{migraphx::gpu::binary_cache_settings{path, false}};
        db_cache.insert(ctx, make_entry("in-a-database"));

        EXPECT(migraphx::fs::is_regular_file(path));
        EXPECT(row_count(path, "cache_v1") == 1);
        EXPECT(entry_files(db_td.path).empty());
        EXPECT(not migraphx::fs::exists(db_td.path / version_dir));
    }
}

// A database that cannot be opened leaves a memory-only cache rather than an error. The parent
// component here is a regular file, so neither creating the directory nor opening the database
// can succeed.
TEST_CASE(unusable_database_degrades_to_memory)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto blocker = td.path / "not_a_dir";
    migraphx::write_buffer(blocker, std::vector<char>(4, 0));
    migraphx::gpu::binary_cache_settings settings{(blocker / "cache.db").string(), false};

    migraphx::gpu::binary_cache cache{settings};
    cache.insert(ctx, make_entry("nowhere"));
    EXPECT(cache.get(ctx, "nowhere").has_value());
    EXPECT(cache.get_stats().reused == 1);

    // Nothing was persisted, so a second cache finds nothing.
    migraphx::gpu::binary_cache reader{settings};
    EXPECT(not reader.get(ctx, "nowhere").has_value());
    EXPECT(reader.get_stats().misses == 1);
}

// A cache path that already holds something other than a cache database is left alone: the
// cache runs from memory, and the file is not overwritten.
TEST_CASE(not_a_database_degrades_to_memory)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    const std::vector<char> garbage(64, 'x');
    migraphx::write_buffer(path, garbage);

    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};
    cache.insert(ctx, make_entry("in-memory"));
    EXPECT(cache.get(ctx, "in-memory").has_value());
    EXPECT(cache.get_stats().reused == 1);
    EXPECT((migraphx::read_buffer(path) == garbage));
}

// A database whose cache table has a different shape, as a future or foreign version might
// leave, cannot be stored into or looked up in, so it is skipped rather than half used.
TEST_CASE(incompatible_schema_degrades_to_memory)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    migraphx::sqlite::write(path).execute("CREATE TABLE cache_v1 (unrelated INTEGER);");

    EXPECT(not migraphx::gpu::sqlite_binary_cache::open(path).has_value());

    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};
    cache.insert(ctx, make_entry("in-memory"));
    EXPECT(cache.get(ctx, "in-memory").has_value());
    EXPECT(row_count(path, "cache_v1") == 0);
}

// Both backends store the same serialized entry, so a cache can move between them.
TEST_CASE(backends_store_identical_bytes)
{
    migraphx::gpu::context ctx;
    auto e = make_entry("interchange");

    migraphx::tmp_dir dir_td{"binary-cache"};
    migraphx::gpu::binary_cache dir_cache{
        migraphx::gpu::binary_cache_settings{dir_path(dir_td), false}};
    dir_cache.insert(ctx, e);
    auto files = entry_files(dir_td.path);
    EXPECT(files.size() == 1);
    auto from_file = migraphx::read_buffer(files.front());
    EXPECT(not from_file.empty());

    migraphx::tmp_dir db_td{"binary-cache"};
    auto path = db_path(db_td);
    migraphx::gpu::binary_cache db_cache{migraphx::gpu::binary_cache_settings{path, false}};
    db_cache.insert(ctx, e);

    auto from_db = db_entries(path);
    EXPECT(from_db.size() == 1);
    EXPECT((from_db.begin()->second == from_file));
}

// A whole compile against each backend has to leave the same kernels behind: the same key
// hashes, each with the same bytes. That makes the choice of backend purely a storage decision.
TEST_CASE(backends_hold_the_same_entries_after_a_compile)
{
    migraphx::tmp_dir dir_td{"binary-cache"};
    migraphx::tmp_dir db_td{"binary-cache"};
    auto path = db_path(db_td);

    auto p_dir = pointwise_program();
    p_dir.compile(migraphx::make_target("gpu"), cache_options(dir_path(dir_td)));
    auto p_db = pointwise_program();
    p_db.compile(migraphx::make_target("gpu"), cache_options(path));

    auto from_dir = dir_entries(dir_td.path);
    auto from_db  = db_entries(path);
    EXPECT(not from_dir.empty());
    EXPECT(from_dir.size() == from_db.size());
    EXPECT((from_dir == from_db));
}

// An entry copied from one backend into the other is a hit there and decodes to the same code,
// so an existing cache can be converted rather than rebuilt. The only translation is the
// version, which a directory names with the short id and a database records in full.
TEST_CASE(entries_move_between_backends)
{
    migraphx::gpu::context ctx;
    const auto& short_version = migraphx::gpu::binary_cache::version_id(true);
    const auto& long_version  = migraphx::gpu::binary_cache::version_id(false);
    auto e                    = make_entry("moving");
    auto expected             = make_code();

    // Directory to database.
    {
        migraphx::tmp_dir dir_td{"binary-cache"};
        migraphx::tmp_dir db_td{"binary-cache"};
        migraphx::gpu::binary_cache writer{
            migraphx::gpu::binary_cache_settings{dir_path(dir_td), false}};
        writer.insert(ctx, e);
        auto files = entry_files(dir_td.path);
        EXPECT(files.size() == 1);

        const auto& file = files.front();
        auto device      = file.parent_path().filename().string();
        auto db          = migraphx::gpu::sqlite_binary_cache::open(db_path(db_td));
        EXPECT(db.has_value());
        db->store(long_version, device, file.stem().string(), e, migraphx::read_buffer(file));

        migraphx::gpu::binary_cache reader{
            migraphx::gpu::binary_cache_settings{db_path(db_td), false}};
        auto found = reader.get(ctx, e.key);
        EXPECT(found.has_value());
        EXPECT(reader.get_stats().hits == 1);
        EXPECT(*found->fragment.get_main_module() == *expected.fragment.get_main_module());
    }

    // Database to directory.
    {
        migraphx::tmp_dir db_td{"binary-cache"};
        migraphx::tmp_dir dir_td{"binary-cache"};
        migraphx::gpu::binary_cache writer{
            migraphx::gpu::binary_cache_settings{db_path(db_td), false}};
        writer.insert(ctx, e);

        auto select = migraphx::sqlite::read(db_path(db_td))
                          .prepare("SELECT version, device, key_hash, entry FROM cache_v1;");
        auto result = select();
        auto rows   = std::vector<migraphx::value>(result.begin(), result.end());
        EXPECT(rows.size() == 1);
        const auto& row = rows.front();
        EXPECT(row.at("version").get_string() == long_version);
        const auto& blob = row.at("entry").get_binary();
        migraphx::gpu::file_binary_cache files{dir_td.path};
        files.store(short_version,
                    row.at("device").get_string(),
                    row.at("key_hash").get_string(),
                    e,
                    std::vector<char>(blob.begin(), blob.end()));

        migraphx::gpu::binary_cache reader{
            migraphx::gpu::binary_cache_settings{dir_path(dir_td), false}};
        auto found = reader.get(ctx, e.key);
        EXPECT(found.has_value());
        EXPECT(reader.get_stats().hits == 1);
        EXPECT(*found->fragment.get_main_module() == *expected.fragment.get_main_module());
    }
}

// Two connections over one database, as two processes compiling against a shared cache would
// have.
TEST_CASE(two_connections_share_a_database)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto path = db_path(td);

    auto a = migraphx::gpu::sqlite_binary_cache::open(path);
    auto b = migraphx::gpu::sqlite_binary_cache::open(path);
    EXPECT(a.has_value());
    EXPECT(b.has_value());

    const std::vector<char> first{'f', 'i', 'r', 's', 't'};
    const std::vector<char> second{'s', 'e', 'c', 'o', 'n', 'd'};

    a->store("v", "dev", "k1", make_entry("k1"), first);
    auto from_b = b->load("v", "dev", "k1");
    EXPECT(from_b.has_value());
    EXPECT((*from_b == first));

    b->store("v", "dev", "k2", make_entry("k2"), second);
    auto from_a = a->load("v", "dev", "k2");
    EXPECT(from_a.has_value());
    EXPECT((*from_a == second));

    EXPECT(not a->load("v", "dev", "absent").has_value());
}

// A table of hashes says nothing about which build wrote it, so each row records the full
// version id. A database has no path length to protect, unlike the directory backend, which
// names its directories with the short one.
TEST_CASE(sqlite_records_the_full_version_id)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};

    cache.insert(ctx, make_entry("one"));
    cache.insert(ctx, make_entry("two"));
    EXPECT(row_count(path, "cache_v1") == 2);

    auto rows = migraphx::sqlite::read(path).execute("SELECT DISTINCT version FROM cache_v1;");
    EXPECT(rows.size() == 1);
    EXPECT(rows.front().at("version") == migraphx::gpu::binary_cache::version_id(false));
}

// The op name, problem and solution are copied into columns only so a cache can be inspected
// with SQL; loads never read them, so nothing else would notice them being wrong.
TEST_CASE(sqlite_records_what_each_entry_was_compiled_for)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    auto e    = make_entry("described");
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};
    cache.insert(ctx, e);

    auto rows = migraphx::sqlite::read(path).execute(
        "SELECT key_hash, op_name, problem, solution FROM cache_v1;");
    EXPECT(rows.size() == 1);
    const auto& row = rows.front();
    EXPECT(row.at("key_hash") == migraphx::md5(e.key));
    EXPECT(row.at("op_name") == e.op_name);
    EXPECT(migraphx::from_json_string(row.at("problem")) == e.problem);
    EXPECT(migraphx::from_json_string(row.at("solution")) == e.solution);
}

// A database that cannot be written to, such as a shared cache installed read-only, still
// serves the entries already in it, and storing into it is quietly skipped.
TEST_CASE(sqlite_read_only_database_still_serves_hits)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    migraphx::gpu::binary_cache_settings settings{path, false};
    {
        migraphx::gpu::binary_cache writer{settings};
        writer.insert(ctx, make_entry("existing"));
    }

    const auto writable = migraphx::fs::perms::owner_write | migraphx::fs::perms::group_write |
                          migraphx::fs::perms::others_write;
    migraphx::fs::permissions(path, writable, migraphx::fs::perm_options::remove);
    // Permissions do not stop root, so what can be checked about stores depends on whether the
    // write protection actually took.
    const bool protected_file = migraphx::sqlite::write(path).read_only();

    migraphx::gpu::binary_cache reader{settings};
    EXPECT(reader.get(ctx, "existing").has_value());
    EXPECT(reader.get_stats().hits == 1);

    reader.insert(ctx, make_entry("new"));
    EXPECT(reader.get(ctx, "new").has_value());
    if(protected_file)
    {
        EXPECT(row_count(path, "cache_v1") == 1);
    }

    // Restored so the temporary directory can be removed, which Windows refuses otherwise.
    migraphx::fs::permissions(
        path, migraphx::fs::perms::owner_write, migraphx::fs::perm_options::add);
}

// version and device separate entries this build may use from entries it may not, so an entry
// stored under one must not be served under another, whichever backend holds it.
TEST_CASE(backends_scope_entries_by_version_and_device)
{
    migraphx::tmp_dir td{"binary-cache"};
    const std::vector<char> blob{'p', 'a', 'y'};
    for(auto& backend : both_backends(td))
    {
        backend.store("v1", "dev1", "k", make_entry("k"), blob);

        EXPECT(backend.load("v1", "dev1", "k").has_value());
        EXPECT(not backend.load("v2", "dev1", "k").has_value());
        EXPECT(not backend.load("v1", "dev2", "k").has_value());
    }
}

// Storing a key twice replaces the entry rather than accumulating or failing. Two processes
// compiling the same kernel is benign for exactly this reason.
TEST_CASE(backends_store_overwrites_in_place)
{
    migraphx::tmp_dir td{"binary-cache"};
    const std::vector<char> replacement{'n', 'e', 'w'};
    for(auto& backend : both_backends(td))
    {
        backend.store("v", "dev", "k", make_entry("k"), {'o', 'l', 'd'});
        backend.store("v", "dev", "k", make_entry("k"), replacement);

        auto got = backend.load("v", "dev", "k");
        EXPECT(got.has_value());
        EXPECT((*got == replacement));
    }
    EXPECT(row_count(db_path(td), "cache_v1") == 1);
    EXPECT(entry_files(td.path / "files").size() == 1);
}

// Publishing an entry goes through a temporary beside it, and nothing of that may be left once
// the entry is in place, including when an existing entry is replaced.
TEST_CASE(file_store_leaves_only_entries_behind)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache_settings settings{dir_path(td), false};

    migraphx::gpu::binary_cache first{settings};
    first.insert(ctx, make_entry("one"));
    first.insert(ctx, make_entry("two"));
    // A second cache stores the same key again, over the file the first one published.
    migraphx::gpu::binary_cache second{settings};
    second.insert(ctx, make_entry("one"));

    std::vector<migraphx::fs::directory_entry> items{
        migraphx::fs::recursive_directory_iterator{td.path},
        migraphx::fs::recursive_directory_iterator{}};
    auto dirs = std::count_if(
        items.begin(), items.end(), [](const auto& item) { return item.is_directory(); });
    // Just the version directory and the device directory inside it, and the two entries.
    EXPECT(dirs == 2);
    EXPECT(entry_files(td.path).size() == 2);
    EXPECT(items.size() == 4);
}

// Storage is opened by the first lookup or insert, not by constructing the cache, since every
// context makes one whether or not it ever compiles anything.
TEST_CASE(storage_is_opened_on_first_use)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};
    EXPECT(not migraphx::fs::exists(path));

    EXPECT(not cache.get(ctx, "absent").has_value());
    EXPECT(migraphx::fs::exists(path));
}

// Inserts made inside a batch are all there once it ends. The database commits them together;
// the directory backend has no batching and stores each one as it comes.
template <class Backend>
static void batched_inserts_are_all_stored()
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = Backend::path(td);
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};
    {
        migraphx::gpu::binary_cache::store_batch batch{cache};
        cache.insert(ctx, make_entry("first"));
        cache.insert(ctx, make_entry("second"));
    }
    EXPECT(Backend::stored(path) == 2);

    migraphx::gpu::binary_cache reader{migraphx::gpu::binary_cache_settings{path, false}};
    EXPECT(reader.get(ctx, "first").has_value());
    EXPECT(reader.get(ctx, "second").has_value());
    EXPECT(reader.get_stats().hits == 2);
}

TEST_CASE_REGISTER(batched_inserts_are_all_stored<directory_backend>);
TEST_CASE_REGISTER(batched_inserts_are_all_stored<database_backend>);

// A batch holds the database's write lock, so another connection must be able to write again as
// soon as it ends.
TEST_CASE(sqlite_batch_releases_the_database)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};
    {
        migraphx::gpu::binary_cache::store_batch batch{cache};
        cache.insert(ctx, make_entry("batched"));
    }

    auto other = migraphx::gpu::sqlite_binary_cache::open(path);
    EXPECT(other.has_value());
    other->store("v", "dev", "k", make_entry("k"), {'x'});
    EXPECT(row_count(path, "cache_v1") == 2);
}

// The backend layer moves opaque bytes and never decodes them, so a payload that is not even
// msgpack still round-trips. Both backends go through the same type-erased wrapper here.
TEST_CASE(backends_round_trip_through_the_wrapper)
{
    migraphx::tmp_dir td{"binary-cache"};
    const std::vector<char> blob{'\0', 'n', 'o', 't', '\0', 'm', 's', 'g', '\xff'};
    auto e = make_entry("opaque");

    for(auto& backend : both_backends(td))
    {
        EXPECT(not backend.load("v", "dev", "k").has_value());
        backend.store("v", "dev", "k", e, blob);
        auto got = backend.load("v", "dev", "k");
        EXPECT(got.has_value());
        EXPECT((*got == blob));
    }
}

TEST_CASE(entry_round_trip)
{
    auto e      = make_entry("some-key");
    auto buffer = migraphx::to_msgpack(migraphx::to_value(e));

    migraphx::gpu::binary_cache::entry loaded;
    migraphx::from_value(migraphx::from_msgpack(buffer), loaded);

    EXPECT(loaded.key == e.key);
    EXPECT(loaded.op_name == e.op_name);
    EXPECT(loaded.solution == e.solution);
    EXPECT(loaded.code.fill_map == e.code.fill_map);
    EXPECT(*loaded.code.fragment.get_main_module() == *e.code.fragment.get_main_module());
}

// The key has to cover everything handed to the compiler, not just the source text. Two
// kernels that differ only in their tensor views or launch bounds must not share an entry.
TEST_CASE(key_covers_more_than_the_source)
{
    migraphx::gpu::context ctx;
    migraphx::shape s{migraphx::shape::float_type, {4, 8}};

    migraphx::gpu::hip_src src;
    src.content             = "// kernel source";
    src.options.inputs      = {s, s};
    src.options.output      = s;
    src.options.global      = 1024;
    src.options.local       = 256;
    src.options.kernel_name = "kernel";
    auto base               = migraphx::gpu::hip_compile_key(ctx, src);

    EXPECT(base == migraphx::gpu::hip_compile_key(ctx, src));

    auto changed_source    = src;
    changed_source.content = "// different source";
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_source) != base);

    auto changed_name                = src;
    changed_name.options.kernel_name = "other_kernel";
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_name) != base);

    auto changed_global           = src;
    changed_global.options.global = 2048;
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_global) != base);

    auto changed_local          = src;
    changed_local.options.local = 128;
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_local) != base);

    auto changed_params = src;
    changed_params.options.emplace_param("-DEXTRA=1");
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_params) != base);

    // Same source, but the generated tensor views differ.
    auto changed_views                   = src;
    changed_views.options.virtual_inputs = {{migraphx::shape::float_type, {32}},
                                            {migraphx::shape::float_type, {32}}};
    EXPECT(migraphx::gpu::hip_compile_key(ctx, changed_views) != base);
}

TEST_CASE(version_id_is_stable)
{
    EXPECT(migraphx::gpu::binary_cache::version_id(true) ==
           migraphx::gpu::binary_cache::version_id(true));
    EXPECT(migraphx::gpu::binary_cache::version_id(false) ==
           migraphx::gpu::binary_cache::version_id(false));
    EXPECT(not migraphx::gpu::binary_cache::version_id(true).empty());
    EXPECT(migraphx::gpu::binary_cache::version_id(true) !=
           migraphx::gpu::binary_cache::version_id(false));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
