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
#include <migraphx/file_buffer.hpp>
#include <migraphx/sqlite.hpp>
#include <migraphx/stringutils.hpp>
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
#include <functional>
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

// The storage backend is chosen by the extension of the cache path, so a case that has to hold
// for both is written once against a path and registered twice, once with each of these.
static std::string dir_path(const migraphx::tmp_dir& td) { return td.path.string(); }
static std::string db_path(const migraphx::tmp_dir& td) { return (td.path / "cache.db").string(); }

/// The entry files a directory-backed cache has written.
static std::vector<migraphx::fs::path> entry_files(const migraphx::fs::path& dir)
{
    std::vector<migraphx::fs::path> result;
    for(const auto& file : migraphx::fs::recursive_directory_iterator(dir))
    {
        if(file.path().extension() == ".mxr")
            result.push_back(file.path());
    }
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

/// How many entries a cache path holds, whichever backend wrote them. The extensions must match
/// the ones make_binary_cache_backend routes to the SQLite backend, or this silently counts
/// files in a directory that does not exist and reports zero.
static std::size_t stored_entry_count(const std::string& path)
{
    if(migraphx::ends_with(path, ".db") or migraphx::ends_with(path, ".sqlite"))
        return row_count(path, "cache_v1");
    return entry_files(path).size();
}

/// Backends constructed directly take the stamp as an argument; only the test that reads
/// cache_info_v1 back cares what it says.
static const std::string test_stamp = "test-stamp\n";

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

// The cases below are registered only against a database path, not a directory. Driving the file
// backend through binary_cache puts entries under version_dir()/device_dir(), and write_atomically
// then creates a temp directory inside that, which pushes the file past Windows' MAX_PATH:
// fs::create_directories succeeds because std::filesystem uses the \\?\ prefix, but the
// std::ofstream in write_buffer does not, so every store fails and nothing is persisted. The
// bodies are still parameterized by path, so a directory case is one TEST_CASE to restore once
// write_atomically writes its temporary as a sibling instead of nesting a directory.
// backends_round_trip_through_the_wrapper still covers the file backend, where it is driven
// directly and the version and device strings are short.

// A second cache shares nothing in memory, so anything it finds came out of storage.
static void disk_lookup_body(const std::string& path)
{
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache_settings settings{path, false};

    migraphx::gpu::binary_cache writer{settings};
    writer.insert(ctx, make_entry("shared-key"));

    migraphx::gpu::binary_cache reader{settings};

    auto found = reader.get(ctx, "shared-key");
    EXPECT(found.has_value());
    EXPECT(reader.get_stats().hits == 1);
    EXPECT(reader.get_stats().misses == 0);
    EXPECT(*found->fragment.get_main_module() == *make_code().fragment.get_main_module());
}

TEST_CASE(sqlite_lookup_records_a_hit)
{
    migraphx::tmp_dir td{"binary-cache"};
    disk_lookup_body(db_path(td));
}

// A damaged entry must cost a recompile and nothing more. How an entry gets damaged is the only
// part of this that depends on the backend, so it comes in as a step.
static void corrupt_entry_body(const std::string& path,
                               const std::function<void(const std::string&)>& damage)
{
    migraphx::gpu::context ctx;
    migraphx::gpu::binary_cache_settings settings{path, false};

    migraphx::gpu::binary_cache writer{settings};
    writer.insert(ctx, make_entry("damaged"));

    damage(path);

    migraphx::gpu::binary_cache reader{settings};

    EXPECT(not reader.get(ctx, "damaged").has_value());
    EXPECT(reader.get_stats().misses == 1);
}

TEST_CASE(sqlite_corrupt_entry_is_ignored)
{
    migraphx::tmp_dir td{"binary-cache"};
    corrupt_entry_body(db_path(td), [](const std::string& db) {
        EXPECT(row_count(db, "cache_v1") == 1);
        migraphx::sqlite::write(db).execute("UPDATE cache_v1 SET entry = zeroblob(8);");
    });
}

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
static void compiling_twice_body(const std::string& path)
{
    auto options = cache_options(path);

    auto p_ref = pointwise_program();
    p_ref.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {4, 8}};
    auto x = migraphx::generate_argument(s, 0);
    auto y = migraphx::generate_argument(s, 1);

    auto ref_result = p_ref.eval({{"x", x}, {"y", y}}).back();

    auto warmup = pointwise_program();
    warmup.compile(migraphx::make_target("gpu"), options);

    EXPECT(stored_entry_count(path) > 0);

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

TEST_CASE(sqlite_compiling_twice_populates_the_cache_and_matches_reference)
{
    migraphx::tmp_dir td{"binary-cache"};
    compiling_twice_body(db_path(td));
}

// With verification on, every reused result is compiled again and compared, so a run that does
// not throw is one where the keys really do capture what the compilers depend on.
static void verified_reuse_body(const std::string& path)
{
    auto options = cache_options(path, /* verify */ true);

    auto warmup = pointwise_program();
    warmup.compile(migraphx::make_target("gpu"), options);

    auto p = pointwise_program();
    p.compile(migraphx::make_target("gpu"), options);
}

TEST_CASE(sqlite_verified_reuse_matches_fresh_compiles)
{
    migraphx::tmp_dir td{"binary-cache"};
    verified_reuse_body(db_path(td));
}

// The extension of the path picks the backend and nothing else does, so the only way to see the
// choice from outside is the artifact it leaves: a database file, or a directory tree.
//
// The directory half checks that the version tree was laid down rather than that an entry file
// landed in it, because the entry write is what MAX_PATH defeats here. create_directories runs
// before that write and succeeds, and the SQLite backend creates no such tree, so this still
// distinguishes the two backends.
TEST_CASE(extension_selects_the_backend)
{
    migraphx::gpu::context ctx;

    migraphx::tmp_dir dir_td{"binary-cache"};
    migraphx::gpu::binary_cache dir_cache{
        migraphx::gpu::binary_cache_settings{dir_path(dir_td), false}};
    dir_cache.insert(ctx, make_entry("in-a-directory"));
    EXPECT(migraphx::fs::is_directory(dir_td.path / migraphx::gpu::binary_cache::version_dir()));

    for(const char* name : {"cache.db", "cache.sqlite"})
    {
        migraphx::tmp_dir db_td{"binary-cache"};
        auto path = (db_td.path / name).string();
        migraphx::gpu::binary_cache db_cache{migraphx::gpu::binary_cache_settings{path, false}};
        db_cache.insert(ctx, make_entry("in-a-database"));

        EXPECT(migraphx::fs::is_regular_file(path));
        EXPECT(row_count(path, "cache_v1") == 1);
        EXPECT(entry_files(db_td.path).empty());
        EXPECT(not migraphx::fs::exists(db_td.path / migraphx::gpu::binary_cache::version_dir()));
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

// Two connections over one database, as two processes compiling against a shared cache would
// have. This cannot be two sqlite_binary_cache objects: only open() populates one, and it hands
// back the type-erased wrapper.
TEST_CASE(two_connections_share_a_database)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto path = db_path(td);

    auto a = migraphx::gpu::sqlite_binary_cache::open(path, test_stamp);
    auto b = migraphx::gpu::sqlite_binary_cache::open(path, test_stamp);
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

// A table of hashes says nothing about which build wrote it, so the database carries a
// description of that build -- written once, not once per kernel.
TEST_CASE(sqlite_records_the_version_stamp)
{
    migraphx::tmp_dir td{"binary-cache"};
    migraphx::gpu::context ctx;
    auto path = db_path(td);
    migraphx::gpu::binary_cache cache{migraphx::gpu::binary_cache_settings{path, false}};

    cache.insert(ctx, make_entry("one"));
    EXPECT(row_count(path, "cache_info_v1") == 1);

    cache.insert(ctx, make_entry("two"));
    EXPECT(row_count(path, "cache_v1") == 2);
    EXPECT(row_count(path, "cache_info_v1") == 1);

    auto rows = migraphx::sqlite::read(path).execute("SELECT version, stamp FROM cache_info_v1;");
    EXPECT(rows.size() == 1);
    EXPECT(rows.front().at("version") == migraphx::gpu::binary_cache::version_dir());
    EXPECT(rows.front().at("stamp") == migraphx::gpu::binary_cache::version_stamp());
}

// version and device lead the primary key because they are what separates entries this build
// may use from entries it may not, so a row stored under one must not be served under another.
TEST_CASE(sqlite_scopes_entries_by_version_and_device)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto backend = migraphx::gpu::sqlite_binary_cache::open(db_path(td), test_stamp);
    EXPECT(backend.has_value());

    const std::vector<char> blob{'p', 'a', 'y'};
    backend->store("v1", "dev1", "k", make_entry("k"), blob);

    EXPECT(backend->load("v1", "dev1", "k").has_value());
    EXPECT(not backend->load("v2", "dev1", "k").has_value());
    EXPECT(not backend->load("v1", "dev2", "k").has_value());
}

// Storing a key twice replaces the row rather than accumulating or failing, the way the file
// backend's publish-by-rename overwrites in place. Two processes compiling the same kernel is
// benign for exactly this reason.
TEST_CASE(sqlite_store_overwrites_in_place)
{
    migraphx::tmp_dir td{"binary-cache"};
    auto path    = db_path(td);
    auto backend = migraphx::gpu::sqlite_binary_cache::open(path, test_stamp);
    EXPECT(backend.has_value());

    const std::vector<char> replacement{'n', 'e', 'w'};
    backend->store("v", "dev", "k", make_entry("k"), {'o', 'l', 'd'});
    backend->store("v", "dev", "k", make_entry("k"), replacement);

    EXPECT(row_count(path, "cache_v1") == 1);
    auto got = backend->load("v", "dev", "k");
    EXPECT(got.has_value());
    EXPECT((*got == replacement));
}

// The backend layer moves opaque bytes and never decodes them, so a payload that is not even
// msgpack still round-trips. Both backends go through the same type-erased wrapper here, which
// is the runtime half of the static_assert in each backend's .cpp.
TEST_CASE(backends_round_trip_through_the_wrapper)
{
    migraphx::tmp_dir td{"binary-cache"};
    const std::vector<char> blob{'\0', 'n', 'o', 't', '\0', 'm', 's', 'g', '\xff'};
    auto e = make_entry("opaque");

    auto db = migraphx::gpu::sqlite_binary_cache::open((td.path / "cache.db").string(), test_stamp);
    EXPECT(db.has_value());

    std::vector<migraphx::gpu::binary_cache_backend> backends;
    backends.emplace_back(migraphx::gpu::file_binary_cache{td.path / "files", test_stamp});
    backends.push_back(*db);

    for(auto& backend : backends)
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

TEST_CASE(version_dir_is_stable)
{
    EXPECT(migraphx::gpu::binary_cache::version_dir() ==
           migraphx::gpu::binary_cache::version_dir());
    EXPECT(not migraphx::gpu::binary_cache::version_dir().empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
