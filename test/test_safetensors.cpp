#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iom/safetensors.hpp"

namespace {

std::string safetensors_test_path() {
    if (const char* path = std::getenv("SAFETENSORS_TEST_DATA")) {
        return path;
    }
    return "/data/models/unsloth/GLM-5.2/model-00001-of-00282.safetensors";
}

}  // namespace

TEST_CASE("SafeTensorsFile exposes tensors from mapped file") {
    const std::string path = safetensors_test_path();
    if (!std::filesystem::exists(path)) {
        MESSAGE("skipping safetensors test data not found: " << path);
        return;
    }

    const iom::SafeTensorsFile file(path);

    CHECK(file.size() > 0);
    REQUIRE(file.keys().size() == file.size());

    const iom::SafeTensorView tensor = file[file.keys().front()];
    CHECK(tensor.nbytes() > 0);
    CHECK(tensor.raw<uint8_t>() != nullptr);

    const auto dtype = tensor.dtype();
    CHECK(static_cast<int>(dtype) >= 0);
}

TEST_CASE("SafeTensorsFile satisfies SafeTensorsStore interface") {
    const std::string path = safetensors_test_path();
    if (!std::filesystem::exists(path)) {
        MESSAGE("skipping safetensors test data not found: " << path);
        return;
    }

    const iom::SafeTensorsFile file(path);
    const iom::SafeTensorsStore& store = file;
    CHECK(store.size() == file.size());
    CHECK(store.keys().size() == file.size());
    const iom::SafeTensorView direct = file[file.keys().front()];
    const iom::SafeTensorView via_iface = store[file.keys().front()];
    CHECK(direct.dtype() == via_iface.dtype());
    CHECK(direct.shape() == via_iface.shape());
    CHECK(direct.nbytes() == via_iface.nbytes());
    CHECK(direct.raw<uint8_t>() == via_iface.raw<uint8_t>());
}

TEST_CASE("SafeTensorsDir exposes tensors from every file in a directory") {
    const std::string path = safetensors_test_path();
    if (!std::filesystem::exists(path)) {
        MESSAGE("skipping safetensors test data not found: " << path);
        return;
    }
    const auto dir_path = std::filesystem::path(path).parent_path();
    if (!std::filesystem::is_directory(dir_path)) {
        MESSAGE("skipping dir test: not a directory: " << dir_path);
        return;
    }

    std::vector<std::filesystem::path> expected_files;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            expected_files.push_back(entry.path());
        }
    }
    std::sort(expected_files.begin(), expected_files.end());
    if (expected_files.size() < 2) {
        MESSAGE("skipping dir test: need >=2 .safetensors shards in " << dir_path);
        return;
    }
    const iom::SafeTensorsDir dir(dir_path.string());
    std::unordered_map<std::string, iom::SafeTensorView> expected;
    size_t expected_total = 0;
    std::vector<std::unique_ptr<iom::SafeTensorsFile>> expected_files_owned;
    expected_files_owned.reserve(expected_files.size());
    for (const auto& fp : expected_files) {
        auto f = std::make_unique<iom::SafeTensorsFile>(fp.string());
        expected_total += f->size();
        for (const auto& key : f->keys()) {
            expected.emplace(key, (*f)[key]);
        }
        expected_files_owned.push_back(std::move(f));
    }
    CHECK(dir.size() == expected_total);
    CHECK(dir.keys().size() == expected.size());

    std::unordered_set<std::string> expected_keys;
    expected_keys.reserve(expected.size());
    for (const auto& [k, v] : expected) {
        expected_keys.insert(k);
    }
    for (const auto& key : dir.keys()) {
        CHECK(expected_keys.count(key) == 1);
    }
    for (const auto& [name, want] : expected) {
        const iom::SafeTensorView got = dir[name];
        CHECK(got.dtype() == want.dtype());
        CHECK(got.shape() == want.shape());
        CHECK(got.nbytes() == want.nbytes());
        const size_t sample = std::min<size_t>(got.nbytes(), 4096);
        CHECK(std::memcmp(got.raw<uint8_t>(), want.raw<uint8_t>(), sample) == 0);
    }

    CHECK_THROWS_AS(dir["no-such-tensor-anywhere"], std::out_of_range);
}

TEST_CASE("SafeTensorsDir rejects missing directory") {
    CHECK_THROWS(iom::SafeTensorsDir("/__iom_no_such_safetensors_dir__"));
}
