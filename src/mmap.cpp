#include "iom/mmap.hpp"

#include <cstring>
#include <stdexcept>

#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace iom {

MappedFile::MappedFile(const std::string& filename, size_t min_size) {
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("open failed for " + filename + ": " +
                                 std::strerror(errno));
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        close(fd_);
        throw std::runtime_error("fstat failed for " + filename + ": " +
                                 std::strerror(errno));
    }

    size_ = static_cast<size_t>(st.st_size);
    if (size_ < min_size) {
        close(fd_);
        throw std::runtime_error("file too small: " + filename);
    }

    void* ptr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed for " + filename + ": " +
                                 std::strerror(errno));
    }

    data_ = static_cast<const uint8_t*>(ptr);
}

MappedFile::~MappedFile() {
    if (data_) {
        munmap(const_cast<uint8_t*>(data_), size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

const uint8_t* MappedFile::data() const {
    return data_;
}

size_t MappedFile::size() const {
    return size_;
}

}  // namespace ec
