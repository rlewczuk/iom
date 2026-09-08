#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "iom/safetensors.hpp"

namespace {

struct TensorEntry {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    std::string payload;
};

// Owns a per-test directory under the system temp path; removed on scope exit.
class TempDir {
public:
    explicit TempDir(const std::string& tag)
        : path_(std::filesystem::temp_directory_path() /
                ("iom-safetensors-" + tag)) {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        if (!std::filesystem::create_directories(path_)) {
            throw std::runtime_error(
                "cannot create test directory: " + path_.string());
        }
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::string deterministic_payload(size_t nbytes, size_t seed) {
    std::string payload(nbytes, '\0');
    for (size_t i = 0; i < nbytes; ++i) {
        payload[i] = static_cast<char>((i * 31 + seed * 7 + 3) & 0xFF);
    }
    return payload;
}

std::string key_name(size_t index) {
    return index < 10 ? "t0" + std::to_string(index)
                      : "t" + std::to_string(index);
}

// Writes a minimal valid safetensors file: 8-byte little-endian header
// length, JSON header, then the concatenated tensor payloads.
std::string write_safetensors_file(const std::filesystem::path& dir,
                                   const std::string& filename,
                                   const std::vector<TensorEntry>& tensors) {
    std::string header = "{";
    std::string payload;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const TensorEntry& tensor = tensors[i];
        if (i > 0) {
            header += ",";
        }
        header += "\"" + tensor.name + "\":{\"dtype\":\"" + tensor.dtype +
                  "\",\"shape\":[";
        for (size_t d = 0; d < tensor.shape.size(); ++d) {
            if (d > 0) {
                header += ",";
            }
            header += std::to_string(tensor.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(payload.size());
        payload += tensor.payload;
        header += "," + std::to_string(payload.size()) + "]}";
    }
    header += "}";

    std::string bytes(8, '\0');
    const auto header_len = static_cast<uint64_t>(header.size());
    for (size_t i = 0; i < 8; ++i) {
        bytes[i] = static_cast<char>((header_len >> (8 * i)) & 0xFF);
    }
    bytes += header;
    bytes += payload;

    const auto path = dir / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "cannot write test safetensors file: " + path.string());
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error(
            "cannot write test safetensors file: " + path.string());
    }
    return path.string();
}

struct RawTensorEntry {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t begin;
    std::size_t end;
};

std::string write_raw_safetensors_file(
    const std::filesystem::path& dir, const std::string& filename,
    const std::vector<RawTensorEntry>& entries, const std::string& payload) {
    std::string header = "{";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const RawTensorEntry& entry = entries[i];
        if (i > 0) {
            header += ",";
        }
        header += "\"" + entry.name + "\":{\"dtype\":\"" + entry.dtype +
                  "\",\"shape\":[";
        for (std::size_t d = 0; d < entry.shape.size(); ++d) {
            if (d > 0) {
                header += ",";
            }
            header += std::to_string(entry.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(entry.begin) + "," +
                  std::to_string(entry.end) + "]}";
    }
    header += "}";

    std::string bytes(8, '\0');
    const auto header_len = static_cast<uint64_t>(header.size());
    for (std::size_t i = 0; i < 8; ++i) {
        bytes[i] = static_cast<char>((header_len >> (8 * i)) & 0xFF);
    }
    bytes += header;
    bytes += payload;

    const auto path = dir / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "cannot write test safetensors file: " + path.string());
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error(
            "cannot write test safetensors file: " + path.string());
    }
    return path.string();
}

TEST_CASE("SafeTensors dtype maps every accepted string to its DataType") {
    const std::pair<std::string, iom::DataType> cases[] = {
        {"BOOL", iom::DataType::BOOL},
        {"U8", iom::DataType::U8},
        {"I8", iom::DataType::I8},
        {"U16", iom::DataType::U16},
        {"I16", iom::DataType::I16},
        {"U32", iom::DataType::U32},
        {"I32", iom::DataType::I32},
        {"U64", iom::DataType::U64},
        {"I64", iom::DataType::I64},
        {"F16", iom::DataType::F16},
        {"BF16", iom::DataType::BF16},
        {"F32", iom::DataType::F32},
        {"F64", iom::DataType::F64},
        {"F8_E5M2", iom::DataType::F8_E5M2},
        {"F8_E4M3", iom::DataType::F8_E4M3FN},
        {"F8_E8M0", iom::DataType::F8_E8M0},
        {"F4", iom::DataType::F4_E2M1},
        {"F6_E2M3", iom::DataType::F6_E2M3},
        {"F6_E3M2", iom::DataType::F6_E3M2},
    };

    TempDir dir("dtype-map");
    std::vector<TensorEntry> tensors;
    std::vector<std::size_t> payload_sizes;
    tensors.reserve(std::size(cases));
    payload_sizes.reserve(std::size(cases));
    for (size_t i = 0; i < std::size(cases); ++i) {
        const std::size_t payload_bytes =
            (iom::detail::leaf_bits(cases[i].second) * 2 + 7) / 8;
        payload_sizes.push_back(payload_bytes);
        tensors.push_back({key_name(i), cases[i].first, {2},
                           deterministic_payload(payload_bytes, i)});
    }
    const auto path = write_safetensors_file(
        dir.path(), "accepted.safetensors", tensors);

    const iom::SafeTensorsFile file(path);
    REQUIRE(file.size() == std::size(cases));
    REQUIRE(file.keys().size() == std::size(cases));

    const std::unordered_set<std::string> keys(file.keys().begin(),
                                               file.keys().end());
    for (size_t i = 0; i < std::size(cases); ++i) {
        CAPTURE(cases[i].first);
        const std::string key = key_name(i);
        REQUIRE(keys.count(key) == 1);

        const iom::SafeTensorView tensor = file[key];
        const std::size_t payload_bytes = payload_sizes[i];
        CHECK(tensor.dtype() == cases[i].second);
        CHECK(tensor.shape() == std::vector<size_t>{2});
        CHECK(tensor.nbytes() == payload_bytes);

        const std::string want = deterministic_payload(payload_bytes, i);
        REQUIRE(tensor.raw<uint8_t>() != nullptr);
        CHECK(std::memcmp(tensor.raw<uint8_t>(), want.data(), payload_bytes) == 0);
    }
}

TEST_CASE("SafeTensorsFile validates complete physical range coverage") {
    {
        TempDir dir("physical-order");
        const std::string first = deterministic_payload(8, 1);
        const std::string second = deterministic_payload(8, 2);
        const auto path = write_raw_safetensors_file(
            dir.path(), "physical-order.safetensors",
            {{"a", "F32", {2}, 8, 16}, {"b", "F32", {2}, 0, 8}},
            second + first);

        const iom::SafeTensorsFile file(path);
        REQUIRE(file.keys() == std::vector<std::string>{"a", "b"});
        CHECK(std::memcmp(file["a"].raw<uint8_t>(), first.data(), first.size()) == 0);
        CHECK(std::memcmp(file["b"].raw<uint8_t>(), second.data(), second.size()) == 0);
    }

    {
        TempDir dir("range-holes");
        const auto throws_range_error = [&](const std::string& filename,
                                            const std::vector<RawTensorEntry>& entries,
                                            const std::string& payload) {
            const auto path =
                write_raw_safetensors_file(dir.path(), filename, entries, payload);
            CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error);
        };
        throws_range_error("leading.safetensors",
                           {{"a", "U8", {3}, 1, 4}}, std::string(4, '\0'));
        throws_range_error("intermediate.safetensors",
                           {{"a", "U8", {4}, 0, 4},
                            {"b", "U8", {4}, 5, 9}},
                           std::string(9, '\0'));
        throws_range_error("trailing.safetensors",
                           {{"a", "U8", {4}, 0, 4}}, std::string(8, '\0'));
        throws_range_error("empty-index.safetensors", {}, "x");
        throws_range_error("overlap.safetensors",
                           {{"a", "U8", {8}, 0, 8},
                            {"b", "U8", {8}, 4, 12}},
                           std::string(12, '\0'));
        throws_range_error("out-of-range.safetensors",
                           {{"a", "U8", {8}, 0, 9}}, std::string(8, '\0'));
    }

    {
        TempDir dir("zero-ranges");
        const auto path = write_raw_safetensors_file(
            dir.path(), "zero-ranges.safetensors",
            {{"a", "U8", {0}, 0, 0}, {"b", "U8", {0}, 0, 0}}, "");
        const iom::SafeTensorsFile file(path);
        REQUIRE(file.keys() == std::vector<std::string>{"a", "b"});
        CHECK(file["a"].nbytes() == 0);
        CHECK(file["b"].nbytes() == 0);
    }

    {
        TempDir dir("payload-validation");
        const auto path = write_raw_safetensors_file(
            dir.path(), "truncated.safetensors",
            {{"a", "F32", {4096, 4096}, 0, 4}},
            deterministic_payload(4, 0));
        CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error);

        std::string message;
        try {
            const iom::SafeTensorsFile file(path);
            (void)file;
        } catch (const std::runtime_error& error) {
            message = error.what();
        }
        CHECK(message.find("payload size mismatch") != std::string::npos);
        CHECK(message.find("a") != std::string::npos);
    }
}

TEST_CASE("SafeTensors dtype rejects unrecognized and nonstandard strings") {
    const std::string rejected[] = {
        "I2", "U2", "I4", "U4",  // sub-byte leaves are not safetensors strings
        "F8_E4M3FN", "F4_E2M1",  // leaf names are not safetensors strings
        "FP16", "BFLOAT16", "f16", "F8_E5M1", "U1", "float32", "",
    };

    for (const auto& dtype : rejected) {
        CAPTURE(dtype);
        TempDir dir("dtype-reject");
        const auto path = write_safetensors_file(
            dir.path(), "rejected.safetensors",
            {{"t", dtype, {1}, std::string(1, '\0')}});
        CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error);
    }
}

TEST_CASE("SafeTensorsFile exposes tensors from a generated file") {
    TempDir dir("file");
    const std::string payload = deterministic_payload(12, 5);
    const auto path = write_safetensors_file(
        dir.path(), "single.safetensors",
        {{"a", "F32", {3, 1}, payload}, {"b", "U8", {4}, "wxyz"}});

    const iom::SafeTensorsFile file(path);
    REQUIRE(file.size() == 2);
    REQUIRE(file.keys().size() == 2);
    CHECK(file.keys()[0] == "a");
    CHECK(file.keys()[1] == "b");

    const iom::SafeTensorView a = file["a"];
    CHECK(a.dtype() == iom::DataType::F32);
    CHECK(a.shape() == std::vector<size_t>{3, 1});
    CHECK(a.nbytes() == 12);
    REQUIRE(a.raw<uint8_t>() != nullptr);
    CHECK(std::memcmp(a.raw<uint8_t>(), payload.data(), 12) == 0);

    const iom::SafeTensorView b = file["b"];
    CHECK(b.dtype() == iom::DataType::U8);
    CHECK(b.shape() == std::vector<size_t>{4});
    CHECK(b.nbytes() == 4);
    CHECK(std::memcmp(b.raw<uint8_t>(), "wxyz", 4) == 0);

    CHECK_THROWS_AS((void)file["missing"], std::out_of_range);
}

TEST_CASE("SafeTensorsFile satisfies SafeTensorsStore interface") {
    TempDir dir("store");
    const auto path = write_safetensors_file(
        dir.path(), "store.safetensors",
        {{"x", "I64", {2}, deterministic_payload(16, 1)}});

    const iom::SafeTensorsFile file(path);
    const iom::SafeTensorsStore& store = file;
    CHECK(store.size() == file.size());
    CHECK(store.keys().size() == file.size());
    const iom::SafeTensorView direct = file["x"];
    const iom::SafeTensorView via_iface = store["x"];
    CHECK(direct.dtype() == via_iface.dtype());
    CHECK(direct.shape() == via_iface.shape());
    CHECK(direct.nbytes() == via_iface.nbytes());
    CHECK(direct.raw<uint8_t>() == via_iface.raw<uint8_t>());
    CHECK_THROWS_AS(store["missing"], std::out_of_range);
}

TEST_CASE("SafeTensorView is non-owning: copy bytes before store destruction") {
    TempDir dir("view-lifetime");
    const std::string payload = deterministic_payload(16, 7);
    const auto path = write_safetensors_file(
        dir.path(), "view-lifetime.safetensors",
        {{"w", "F32", {4}, payload}});

    std::vector<std::uint8_t> copied;
    {
        const iom::SafeTensorsFile file(path);
        const iom::SafeTensorView view = file["w"];
        REQUIRE(view.nbytes() == payload.size());
        copied.resize(view.nbytes());
        std::memcpy(copied.data(), view.raw<std::uint8_t>(), view.nbytes());
    }

    REQUIRE(copied.size() == payload.size());
    CHECK(std::memcmp(copied.data(), payload.data(), payload.size()) == 0);
}

TEST_CASE("SafeTensorsDir rejects duplicate keys across shards") {
    {
        TempDir dir("duplicate-shards");
        const auto shard_a = write_safetensors_file(
            dir.path(), "a.safetensors",
            {{"k", "F32", {1}, "AAAA"}});
        const auto shard_b = write_safetensors_file(
            dir.path(), "b.safetensors",
            {{"k", "F32", {1}, "BBBB"}});

        CHECK_THROWS_AS((void)iom::SafeTensorsDir(dir.path().string()),
                        std::runtime_error);

        std::string message;
        try {
            const iom::SafeTensorsDir shards(dir.path().string());
            (void)shards;
        } catch (const std::runtime_error& error) {
            message = error.what();
        }
        CHECK(message.find("duplicate tensor key across shards") !=
              std::string::npos);
        CHECK(message.find("'k'") != std::string::npos);
        CHECK(message.find(shard_a) != std::string::npos);
        CHECK(message.find(shard_b) != std::string::npos);
    }

    {
        TempDir dir("disjoint-shards");
        write_safetensors_file(
            dir.path(), "a.safetensors",
            {{"a1", "F32", {1}, "A001"}});
        write_safetensors_file(
            dir.path(), "b.safetensors",
            {{"b1", "F32", {1}, "B001"}, {"b2", "F32", {1}, "B002"}});
        write_safetensors_file(
            dir.path(), "c.safetensors",
            {{"c1", "F32", {1}, "C001"}});

        const iom::SafeTensorsDir shards(dir.path().string());
        REQUIRE(shards.size() == 4);
        REQUIRE(shards.keys().size() == 4);
        CHECK(shards.keys() ==
              std::vector<std::string>{"a1", "b1", "b2", "c1"});
        CHECK(shards.keys()[0] == "a1");
        CHECK(shards.keys()[1] == "b1");
        CHECK(shards.keys()[2] == "b2");
        CHECK(shards.keys()[3] == "c1");

        const std::pair<const char*, const char*> expected[] = {
            {"a1", "A001"}, {"b1", "B001"}, {"b2", "B002"}, {"c1", "C001"}};
        for (const auto& [key, payload] : expected) {
            const auto tensor = shards[key];
            REQUIRE(tensor.nbytes() == 4);
            REQUIRE(tensor.raw<uint8_t>() != nullptr);
            CHECK(std::memcmp(tensor.raw<uint8_t>(), payload, 4) == 0);
        }
    }

    {
        TempDir dir("duplicate-three-way");
        const auto shard_a = write_safetensors_file(
            dir.path(), "a.safetensors",
            {{"k", "F32", {1}, "AAAA"}});
        const auto shard_b = write_safetensors_file(
            dir.path(), "b.safetensors",
            {{"k", "F32", {1}, "BBBB"}});
        write_safetensors_file(
            dir.path(), "c.safetensors",
            {{"k", "F32", {1}, "CCCC"}});

        CHECK_THROWS_AS((void)iom::SafeTensorsDir(dir.path().string()),
                        std::runtime_error);

        std::string message;
        try {
            const iom::SafeTensorsDir shards(dir.path().string());
            (void)shards;
        } catch (const std::runtime_error& error) {
            message = error.what();
        }
        CHECK(message.find("'k'") != std::string::npos);
        CHECK(message.find(shard_a) != std::string::npos);
        CHECK(message.find(shard_b) != std::string::npos);
    }
}

TEST_CASE("SafeTensorsDir exposes tensors from every file in a directory") {
    TempDir dir("shards");
    const auto shard_a = write_safetensors_file(
        dir.path(), "a.safetensors",
        {{"a1", "BOOL", {2}, std::string("\1\0", 2)}});
    const auto shard_b = write_safetensors_file(
        dir.path(), "b.safetensors",
        {{"b1", "F16", {2, 2}, deterministic_payload(8, 2)},
         {"b2", "F4", {1}, std::string(1, '\xAB')}});
    // Non-safetensors files must be ignored.
    { std::ofstream notes(dir.path() / "notes.txt"); notes << "not a shard"; }

    const iom::SafeTensorsDir shards(dir.path().string());
    CHECK(shards.size() == 3);
    CHECK(shards.keys().size() == 3);

    std::vector<std::unique_ptr<iom::SafeTensorsFile>> expected_files;
    expected_files.push_back(std::make_unique<iom::SafeTensorsFile>(shard_a));
    expected_files.push_back(std::make_unique<iom::SafeTensorsFile>(shard_b));

    std::unordered_set<std::string> expected_keys;
    for (const auto& file : expected_files) {
        for (const auto& key : file->keys()) {
            expected_keys.insert(key);
        }
    }
    REQUIRE(expected_keys.size() == 3);
    for (const auto& key : shards.keys()) {
        CHECK(expected_keys.count(key) == 1);
    }

    for (const auto& file : expected_files) {
        for (const auto& key : file->keys()) {
            const iom::SafeTensorView want = (*file)[key];
            const iom::SafeTensorView got = shards[key];
            CHECK(got.dtype() == want.dtype());
            CHECK(got.shape() == want.shape());
            CHECK(got.nbytes() == want.nbytes());
            REQUIRE(got.raw<uint8_t>() != nullptr);
            CHECK(std::memcmp(got.raw<uint8_t>(), want.raw<uint8_t>(),
                              want.nbytes()) == 0);
        }
    }

    CHECK_THROWS_AS((void)shards["no-such-tensor-anywhere"], std::out_of_range);
}

TEST_CASE("SafeTensorsDir rejects missing directory") {
    CHECK_THROWS(iom::SafeTensorsDir("/__iom_no_such_safetensors_dir__"));
}

}  // namespace
