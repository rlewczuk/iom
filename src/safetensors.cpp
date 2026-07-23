#include "iom/safetensors.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

namespace iom {
namespace {

uint64_t read_le_u64(const uint8_t* data) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (i * 8);
    }
    return value;
}

DataType parse_dtype(const std::string& dtype) {
    if (dtype == "BOOL") return DataType::BOOL;
    if (dtype == "U8") return DataType::U8;
    if (dtype == "I8") return DataType::I8;
    if (dtype == "U16") return DataType::U16;
    if (dtype == "I16") return DataType::I16;
    if (dtype == "U32") return DataType::U32;
    if (dtype == "I32") return DataType::I32;
    if (dtype == "U64") return DataType::U64;
    if (dtype == "I64") return DataType::I64;
    if (dtype == "F16") return DataType::F16;
    if (dtype == "BF16") return DataType::BF16;
    if (dtype == "F32") return DataType::F32;
    if (dtype == "F64") return DataType::F64;
    if (dtype == "F8_E5M2") return DataType::F8_E5M2;
    if (dtype == "F8_E4M3") return DataType::F8_E4M3;
    if (dtype == "F8_E8M0") return DataType::F8_E8M0;
    if (dtype == "F6_E2M3") return DataType::F6_E2M3;
    if (dtype == "F6_E3M2") return DataType::F6_E3M2;
    if (dtype == "F4") return DataType::F4;

    throw std::runtime_error("unsupported safetensors dtype: " + dtype);
}

size_t json_size(const nlohmann::json& value, const char* field) {
    if (!value.is_number_unsigned()) {
        throw std::runtime_error(std::string("invalid safetensors ") + field);
    }
    const uint64_t raw = value.get<uint64_t>();
    if (raw > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string("safetensors ") + field + " is too large");
    }
    return static_cast<size_t>(raw);
}

}  // namespace

SafeTensorView::SafeTensorView(DataType dtype, std::vector<size_t> shape,
                               const uint8_t* data, size_t nbytes)
    : dtype_(dtype), shape_(std::move(shape)), data_(data), nbytes_(nbytes) {
}

const std::vector<size_t>& SafeTensorView::shape() const {
    return shape_;
}

DataType SafeTensorView::dtype() const {
    return dtype_;
}

size_t SafeTensorView::nbytes() const {
    return nbytes_;
}

SafeTensorsFile::SafeTensorsFile(const std::string& filename)
    : file_(filename, 8) {
    const uint64_t header_len_u64 = read_le_u64(file_.data());
    if (header_len_u64 > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("safetensors header is too large: " + filename);
    }

    const size_t header_len = static_cast<size_t>(header_len_u64);
    if (header_len > file_.size() - 8) {
        throw std::runtime_error("safetensors header exceeds file size: " + filename);
    }

    const uint8_t* header_begin = file_.data() + 8;
    const uint8_t* data_begin = header_begin + header_len;
    const size_t data_size = file_.size() - 8 - header_len;

    const auto header = nlohmann::json::parse(
        reinterpret_cast<const char*>(header_begin),
        reinterpret_cast<const char*>(data_begin));
    if (!header.is_object()) {
        throw std::runtime_error("invalid safetensors header: " + filename);
    }

    for (const auto& [name, tensor] : header.items()) {
        if (name == "__metadata__") {
            continue;
        }
        if (!tensor.is_object()) {
            throw std::runtime_error("invalid safetensors tensor entry: " + name);
        }

        const auto dtype_it = tensor.find("dtype");
        const auto shape_it = tensor.find("shape");
        const auto offsets_it = tensor.find("data_offsets");
        if (dtype_it == tensor.end() || !dtype_it->is_string() ||
            shape_it == tensor.end() || !shape_it->is_array() ||
            offsets_it == tensor.end() || !offsets_it->is_array() || offsets_it->size() != 2) {
            throw std::runtime_error("invalid safetensors tensor fields: " + name);
        }

        std::vector<size_t> shape;
        shape.reserve(shape_it->size());
        for (const auto& dim : *shape_it) {
            shape.push_back(json_size(dim, "shape dimension"));
        }

        const size_t begin = json_size((*offsets_it)[0], "data offset");
        const size_t end = json_size((*offsets_it)[1], "data offset");
        if (begin > end || end > data_size) {
            throw std::runtime_error("safetensors data offset out of range: " + name);
        }

        keys_.push_back(name);
        tensors_.emplace(name, SafeTensorView(parse_dtype(dtype_it->get<std::string>()),
                                             std::move(shape), data_begin + begin, end - begin));
    }
}

SafeTensorView SafeTensorsFile::operator[](const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::out_of_range("safetensors tensor not found: " + name);
    }
    return it->second;
}

size_t SafeTensorsFile::size() const {
    return tensors_.size();
}

const std::vector<std::string>& SafeTensorsFile::keys() const {
    return keys_;
}

}  // namespace ec
