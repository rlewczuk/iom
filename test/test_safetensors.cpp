#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

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
