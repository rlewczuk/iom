#include <doctest/doctest.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "iom/mmap.hpp"

namespace {

// Builds a unique temp path under /tmp for a single test instance.
std::string temp_path(const char* suffix) {
    static unsigned long counter = 0;
    return "/tmp/ec_mmap_test_" + std::to_string(++counter) + "_" + suffix;
}

// Writes the given bytes to a file and returns the path.
std::string write_temp_file(const char* suffix, const std::vector<uint8_t>& bytes) {
    std::string path = temp_path(suffix);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    return path;
}

void remove_temp_file(const std::string& path) {
    std::remove(path.c_str());
}

}  // namespace

TEST_CASE("MappedFile exposes mapped bytes") {
    const std::vector<uint8_t> payload = {'h', 'e', 'l', 'l', 'o', '!', '!', '!'};
    std::string path = write_temp_file("happy.bin", payload);

    {
        iom::MappedFile mf(path);
        CHECK_EQ(mf.size(), payload.size());
        CHECK_EQ(std::memcmp(mf.data(), payload.data(), payload.size()), 0);
    }

    remove_temp_file(path);
}

// doctest's expression decomposition mis-parses constructor-call expressions
// like `ec::MappedFile(path)` inside CHECK_THROWS_AS, so we use a plain try/catch.
template <typename Exc>
bool construct_throws(const std::string& path) {
    try {
        iom::MappedFile mf(path);
    } catch (const Exc&) {
        return true;
    }
    return false;
}

TEST_CASE("MappedFile throws on missing file") {
    std::string path = temp_path("missing.bin");
    CHECK(construct_throws<std::runtime_error>(path));
}

TEST_CASE("MappedFile rejects files smaller than 8 bytes") {
    const std::vector<uint8_t> small = {'a', 'b', 'c'};
    std::string path = write_temp_file("small.bin", small);

    CHECK(construct_throws<std::runtime_error>(path));

    remove_temp_file(path);
}
