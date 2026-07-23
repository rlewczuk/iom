#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "iom/iom.hpp"
#include "iom/mmap.hpp"

namespace iom {

class SafeTensorView {
public:
    SafeTensorView() = default;
    SafeTensorView(DataType dtype, std::vector<size_t> shape,
                   const uint8_t* data, size_t nbytes);

    [[nodiscard]]
    const std::vector<size_t>& shape() const;

    template <typename DTYPE>
    [[nodiscard]]
    const DTYPE* raw() const {
        return reinterpret_cast<const DTYPE*>(data_);
    }

    [[nodiscard]]
    DataType dtype() const;

    [[nodiscard]]
    size_t nbytes() const;

private:
    DataType dtype_ = DataType::U8;
    std::vector<size_t> shape_;
    const uint8_t* data_ = nullptr;
    size_t nbytes_ = 0;
};

class SafeTensorsFile {
public:
    explicit SafeTensorsFile(const std::string& filename);

    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    [[nodiscard]]
    SafeTensorView operator[](const std::string& name) const;

    [[nodiscard]]
    size_t size() const;

    [[nodiscard]]
    const std::vector<std::string>& keys() const;

private:
    MappedFile file_;
    std::unordered_map<std::string, SafeTensorView> tensors_;
    std::vector<std::string> keys_;
};

}  // namespace ec
