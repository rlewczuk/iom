#pragma once

// Fixed-purpose reader for the independent synthetic TinyLlama corpus.  This
// is test infrastructure only: it materializes the same ordinary config,
// tokenizer, and SafeTensors bytes consumed by the production model loader.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "../model_loading_fixture.hpp"
#include "../tokenizer_fixture.hpp"

namespace iom_model_reference {

using iom_model_loading::SafetensorsEntry;
using iom_model_loading::TempDir;
using iom_model_loading::element_count;
using iom_model_loading::required_weight_entries;
using iom_model_loading::write_config;
using iom_model_loading::write_file;
using iom_model_loading::write_safetensors_file;
using iom_tokenizer_test::tokenizer_config;
using iom_tokenizer_test::write_tokenizer;

inline std::uint8_t hex_digit(char value) {
    if (value >= '0' && value <= '9') {
        return static_cast<std::uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<std::uint8_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
        return static_cast<std::uint8_t>(value - 'A' + 10);
    }
    throw std::invalid_argument("synthetic reference BF16 payload is not hexadecimal");
}

inline std::string decode_bf16_payload(const nlohmann::json& record) {
    const std::string encoded = record.at("bf16_le_hex").get<std::string>();
    if (encoded.size() % 4 != 0) {
        throw std::invalid_argument(
                "synthetic reference BF16 payload has an odd element length");
    }
    const std::vector<std::size_t> shape = record.at("shape").get<std::vector<std::size_t>>();
    if (encoded.size() != element_count(shape) * 4) {
        throw std::invalid_argument(
                "synthetic reference BF16 payload size does not match its shape");
    }
    std::string payload(encoded.size() / 2, '\0');
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<char>(
                (hex_digit(encoded[index * 2]) << 4) | hex_digit(encoded[index * 2 + 1]));
    }
    return payload;
}

inline std::size_t next_fixture_serial() {
    static std::size_t serial = 0;
    return serial++;
}

inline void validate_model_record(const nlohmann::json& model) {
    if (!model.is_object() || !model.contains("id") || !model.contains("config")
            || !model.contains("weights") || !model.at("id").is_string()) {
        throw std::invalid_argument("synthetic reference model record is malformed");
    }
    const std::string id = model.at("id").get<std::string>();
    const auto path_safe = [](char value) {
        return (value >= 'A' && value <= 'Z')
                || (value >= 'a' && value <= 'z')
                || (value >= '0' && value <= '9')
                || value == '.' || value == '_' || value == '-';
    };
    if (id.empty() || !std::all_of(id.begin(), id.end(), path_safe)) {
        throw std::invalid_argument(
                "synthetic reference model id is not path-safe");
    }
    if (!model.at("config").is_object() || !model.at("weights").is_array()) {
        throw std::invalid_argument("synthetic reference model config or weights is malformed");
    }
}

inline std::string fixture_directory_name(const nlohmann::json& model) {
    validate_model_record(model);
    return "synthetic-reference-" + model.at("id").get<std::string>() + "-"
            + std::to_string(next_fixture_serial());
}

/**
 * A materialized synthetic distribution.  The directory owns every byte used
 * by model loading and must outlive any source, model, session, or queue drain
 * that consumes it.
 */
class SyntheticFixture final {
public:
    explicit SyntheticFixture(const nlohmann::json& model)
        : directory(fixture_directory_name(model)) {
        const nlohmann::json config = model.at("config");
        write_config(directory.path(), config);

        std::vector<SafetensorsEntry> entries = required_weight_entries(config);
        const nlohmann::json& records = model.at("weights");
        if (records.size() != entries.size()) {
            throw std::invalid_argument(
                    "synthetic reference weight count does not match the config");
        }
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const nlohmann::json& record = records.at(index);
            if (!record.is_object() || record.at("name").get<std::string>() != entries[index].name
                    || record.at("shape").get<std::vector<std::size_t>>()
                            != entries[index].shape
                    || record.at("dtype").get<std::string>() != "BF16") {
                throw std::invalid_argument(
                        "synthetic reference weight identity or shape mismatch");
            }
            entries[index].payload = decode_bf16_payload(record);
        }
        write_safetensors_file(directory.path(), "model.safetensors", entries);
        write_tokenizer(directory.path());
        write_file(directory.path(), "tokenizer_config.json", tokenizer_config().dump());
    }

    SyntheticFixture(const SyntheticFixture&) = delete;
    SyntheticFixture& operator=(const SyntheticFixture&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return directory.path();
    }

    // Public owner ordering makes it natural for a test/session fixture to
    // retain this object next to the session and drain it after all work.
    TempDir directory;
};
inline nlohmann::json load_synthetic_corpus() {
#ifdef IOM_MODEL_REFERENCE_DIR
    const std::filesystem::path path =
            std::filesystem::path(IOM_MODEL_REFERENCE_DIR) / "synthetic_reference.json";
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
                "cannot open synthetic reference corpus: " + path.string());
    }
    try {
        nlohmann::json corpus;
        input >> corpus;
        return corpus;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
                "cannot parse synthetic reference corpus " + path.string() + ": "
                + error.what());
    }
#else
    throw std::runtime_error(
            "IOM_MODEL_REFERENCE_DIR must be defined for the synthetic reference corpus");
#endif
}

inline std::size_t tensor_element_count(const nlohmann::json& shape,
                                        std::string_view context) {
    if (!shape.is_array() || shape.empty()) {
        throw std::invalid_argument(
                std::string(context) + " shape must be a non-empty array");
    }
    std::size_t count = 1;
    for (const nlohmann::json& value : shape) {
        if ((!value.is_number_unsigned() && !value.is_number_integer())
                || value.get<std::int64_t>() <= 0) {
            throw std::invalid_argument(
                    std::string(context) + " shape has a non-positive dimension");
        }
        const std::size_t dimension = value.get<std::size_t>();
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                    std::string(context) + " shape element count overflows");
        }
        count *= dimension;
    }
    return count;
}

/** Check every observed value against the independent per-value interval. */
inline void check_values(std::span<const float> actual,
                         const nlohmann::json& tensor_record,
                         double atol,
                         double rtol) {
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0.0 || rtol < 0.0) {
        throw std::invalid_argument("synthetic reference comparison tolerances are invalid");
    }
    if (!tensor_record.is_object() || !tensor_record.contains("shape")
            || !tensor_record.contains("values")) {
        throw std::invalid_argument("synthetic reference tensor record is malformed");
    }
    const std::size_t count =
            tensor_element_count(tensor_record.at("shape"), "synthetic reference tensor");
    const nlohmann::json& expected = tensor_record.at("values");
    if (!expected.is_array() || expected.size() != count) {
        throw std::invalid_argument(
                "synthetic reference tensor payload length does not match its shape");
    }
    if (actual.size() != count) {
        throw std::invalid_argument(
                "synthetic reference actual payload length does not match its shape");
    }
    for (std::size_t index = 0; index < count; ++index) {
        const float observed = actual[index];
        if (!std::isfinite(observed)) {
            throw std::invalid_argument(
                    "synthetic reference actual payload contains a non-finite value");
        }
        if (!expected[index].is_number()) {
            throw std::invalid_argument(
                    "synthetic reference expected payload contains a non-number");
        }
        const double reference = expected[index].get<double>();
        if (!std::isfinite(reference)) {
            throw std::invalid_argument(
                    "synthetic reference expected payload contains a non-finite value");
        }
        const double bound = atol + rtol * std::abs(reference);
        if (std::abs(static_cast<double>(observed) - reference) > bound) {
            throw std::runtime_error(
                    "synthetic reference value mismatch at index "
                    + std::to_string(index));
        }
    }
}

/**
 * Return whether one vocabulary vector has a strictly stable greedy winner.
 * Equal logits use the lowest ID for the reported winner, but equality of
 * tolerance intervals is deliberately unstable.
 */
inline bool margin_stable(std::span<const float> logits,
                          double atol,
                          double rtol) {
    if (logits.empty() || !std::isfinite(atol) || !std::isfinite(rtol)
            || atol < 0.0 || rtol < 0.0) {
        return false;
    }
    for (const float value : logits) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    std::size_t winner = 0;
    for (std::size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > logits[winner]) {
            winner = index;
        }
    }
    double runner = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (index != winner) {
            runner = std::max(runner, static_cast<double>(logits[index]));
        }
    }
    const double winner_value = logits[winner];
    const double winner_error = atol + rtol * std::abs(winner_value);
    const double runner_error = atol + rtol * std::abs(runner);
    return winner_value - winner_error > runner + runner_error;
}

}  // namespace iom_model_reference
