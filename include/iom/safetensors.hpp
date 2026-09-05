#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "iom/iom.hpp"
#include "iom/mmap.hpp"

namespace iom {

/**
 * Copyable, non-owning byte view into mapped bytes owned by the enclosing
 * SafeTensorsStore. raw<T>() borrows those bytes and nbytes() describes their
 * borrowed size; the store must outlive this view and every raw pointer
 * obtained from it. Copying a view copies only its pointer and metadata and
 * does not retain the store. Destroying the store invalidates the view's
 * mapped bytes and all raw pointers obtained from it.
 */

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

/**
 * A concrete SafeTensorsStore owns the backing mapping resources for the
 * views it returns and must outlive every SafeTensorView and raw pointer it
 * produced. Destroying the store invalidates all such views and raw pointers;
 * copying a returned view does not extend this lifetime.
 */

class SafeTensorsStore {
public:
    virtual ~SafeTensorsStore() = default;

    [[nodiscard]]
    virtual SafeTensorView operator[](const std::string& name) const = 0;

    [[nodiscard]]
    virtual size_t size() const = 0;

    [[nodiscard]]
    virtual const std::vector<std::string>& keys() const = 0;
};

namespace detail {

class SafeTensorsStoreBase {
public:
    bool insert(const std::string& name, const SafeTensorView& view) {
        auto [it, inserted] = tensors_.emplace(name, view);
        if (inserted) {
            keys_.push_back(name);
        }
        return inserted;
    }

    const SafeTensorView& at(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::out_of_range("safetensors tensor not found: " + name);
        }
        return it->second;
    }

    size_t size() const noexcept {
        return tensors_.size();
    }

    const std::vector<std::string>& keys() const noexcept {
        return keys_;
    }

private:
    std::unordered_map<std::string, SafeTensorView> tensors_;
    std::vector<std::string> keys_;
};

}  // namespace detail

/**
 * SafeTensorsFile owns its mapped bytes through MappedFile file_. It must
 * outlive every SafeTensorView and raw pointer produced from it; destroying
 * the file invalidates them. A view returned by value is non-owning and does
 * not retain file_.
 */

class SafeTensorsFile : public SafeTensorsStore {
public:
    explicit SafeTensorsFile(const std::string& filename);

    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    [[nodiscard]]
    SafeTensorView operator[](const std::string& name) const override { return base_.at(name); }

    [[nodiscard]]
    size_t size() const override { return base_.size(); }

    [[nodiscard]]
    const std::vector<std::string>& keys() const override { return base_.keys(); }

private:
    MappedFile file_;
    iom::detail::SafeTensorsStoreBase base_;
};

/**
 * SafeTensorsDir owns shard mappings through
 * std::vector<std::unique_ptr<SafeTensorsFile>> files_. The directory store
 * must outlive every SafeTensorView and raw pointer produced from it;
 * destroying the directory invalidates them. Its by-value operator[] returns
 * a non-owning SafeTensorView and does not extend the selected
 * SafeTensorsFile lifetime.
 */

class SafeTensorsDir : public SafeTensorsStore {
public:
    explicit SafeTensorsDir(const std::string& dirname);

    SafeTensorsDir(const SafeTensorsDir&) = delete;
    SafeTensorsDir& operator=(const SafeTensorsDir&) = delete;

    [[nodiscard]]
    SafeTensorView operator[](const std::string& name) const override { return base_.at(name); }

    [[nodiscard]]
    size_t size() const override { return base_.size(); }

    [[nodiscard]]
    const std::vector<std::string>& keys() const override { return base_.keys(); }

private:
    std::vector<std::unique_ptr<SafeTensorsFile>> files_;
    iom::detail::SafeTensorsStoreBase base_;
    std::unordered_map<std::string, std::string> existing_shard_paths_;
};

}  // namespace ec
