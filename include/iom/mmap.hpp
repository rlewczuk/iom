#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace iom {

class MappedFile {
public:
    explicit MappedFile(const std::string& filename, size_t min_size = 8);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    [[nodiscard]]
    const uint8_t* data() const;

    [[nodiscard]]
    size_t size() const;

private:
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace ec
