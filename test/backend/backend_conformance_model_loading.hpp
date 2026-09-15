#pragma once

// Backend-neutral conformance harness: shared model loading and weight
// realization scenario (change 0006-tinyllama / 02-configuration-weight-loading).
//
// Loads independent synthetic TinyLlama checkpoints through the production
// configuration and mapped-source API, creates every destination from the
// published selected specification, preflights the reusable transfer
// requirement of the complete ordered binding, realizes that binding
// synchronously, and reads every destination back with real `TensorView`
// transfers compared bit-for-bit against the fixture bytes. Only a backend
// driver creates the runtime devices and allocators; this scenario owns the
// fixture directory, the source, the destination owners, and any caller
// workspace through ordinary public APIs under RAII. It contains no
// backend-kind switch, no accelerator header, no production failure-injection
// seam, and no second inventory generator, so the CPU, CUDA, ROCm, SYCL, and
// TTNN drivers call it unchanged, in that fixed integration order.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "iom/device.hpp"
#include "iom/model.hpp"
#include "iom/tensor.hpp"
#include "model_loading_fixture.hpp"

namespace iom_conformance {
namespace model_loading {

// ---------------------------------------------------------------------------
// Caller-owned fixture and destination binding.
// ---------------------------------------------------------------------------

// One fixture directory name per selected device and case. The tag carries the
// reported backend kind and ordinal of the selected device so two backends'
// conformance targets may run concurrently without sharing a temporary
// directory; no behavior depends on those values.
inline std::string fixture_tag(const ConformanceDevices& devices,
                               std::string_view name) {
    return "conformance-model-loading-" +
           std::to_string(
                   static_cast<unsigned>(devices.candidate.backend_kind())) +
           "-" +
           std::to_string(devices.candidate.backend_device()) + "-" +
           std::string(name);
}

// Writes `config` and its complete required checkpoint into a fresh fixture
// directory and publishes the mapped source.
inline std::unique_ptr<iom::ModelSource> load_checkpoint(
        const std::filesystem::path& directory, const nlohmann::json& config,
        const std::vector<iom_model_loading::SafetensorsEntry>& entries) {
    iom_model_loading::write_config(directory, config);
    iom_model_loading::write_safetensors_file(directory, "model.safetensors",
                                              entries);
    std::unique_ptr<iom::ModelSource> source =
            iom::load_tinyllama_safetensors(directory);
    REQUIRE(source != nullptr);
    return source;
}

// One caller-owned destination binding: exactly one independent owner per
// published inventory entry, in inventory order, plus the exact borrowed
// pointer list the fixed upload ABI takes. The owners are what keeps a
// realized weight readable, so a case can release the source and keep reading
// the same binding.
struct DestinationBinding {
    std::vector<std::unique_ptr<iom::Tensor>> owners;
    std::vector<iom::Tensor*> pointers;

    // The exact borrowed destination list the fixed upload ABI takes.
    [[nodiscard]] std::span<iom::Tensor* const> span() const noexcept {
        return std::span<iom::Tensor* const>{pointers};
    }
};

// Creates one destination owner per published inventory entry from the
// source's own selected specification, in exact inventory order.
inline DestinationBinding create_binding(iom::Device& device,
                                         const iom::ModelSource& source) {
    const std::span<const iom::ModelWeightInfo> weights = source.weights();
    DestinationBinding binding;
    binding.owners.reserve(weights.size());
    for (std::size_t index = 0; index < weights.size(); ++index) {
        binding.owners.push_back(
                device.create_tensor(source.tensor_spec(index)));
    }
    binding.pointers = iom_model_loading::destination_pointers(binding.owners);
    return binding;
}

// ---------------------------------------------------------------------------
// Sentinel seeding and observation.
// ---------------------------------------------------------------------------

// The sentinel byte of one destination: deterministic, distinct between two
// destinations, and reachable by neither an uploaded fixture payload nor a
// partially copied one, so an unintended write is observable in the exact
// destination bytes.
inline std::byte sentinel_byte(std::size_t index) noexcept {
    return static_cast<std::byte>(0xE1u + (index % 29u));
}

// Seeds every destination of one binding with its own sentinel through a real
// host transfer and returns the exact expected bytes per index.
inline std::vector<std::vector<std::byte>> seed_sentinels(
        DestinationBinding& binding) {
    std::vector<std::vector<std::byte>> sentinels;
    sentinels.reserve(binding.owners.size());
    for (std::size_t index = 0; index < binding.owners.size(); ++index) {
        const std::size_t nbytes =
                binding.owners[index]->view().spec().logical_nbytes();
        sentinels.emplace_back(nbytes, sentinel_byte(index));
        iom_conformance::copy_from_host(binding.owners[index]->view(),
                                       sentinels.back());
    }
    return sentinels;
}

// Requires every destination to still hold exactly its own seeded sentinel, so
// a rejected call wrote no byte of any destination and published nothing.
inline void require_sentinels(
        const DestinationBinding& binding,
        const std::vector<std::vector<std::byte>>& sentinels,
        std::string_view context) {
    REQUIRE(binding.owners.size() == sentinels.size());
    for (std::size_t index = 0; index < binding.owners.size(); ++index) {
        CAPTURE(index);
        const std::vector<std::byte> actual =
                iom_conformance::read_logical(binding.owners[index]->view());
        REQUIRE(actual.size() == sentinels[index].size());
        CHECK_MESSAGE(
                std::memcmp(actual.data(), sentinels[index].data(),
                            actual.size()) == 0,
                context << ": rejected call wrote destination " << index);
    }
}

// ---------------------------------------------------------------------------
// Requirement query, scratch provisioning, and byte verification.
// ---------------------------------------------------------------------------

// The real maximum serial per-owner host-transfer requirement of one binding,
// read from the destination views themselves, so the reported preflight is
// compared with the values a backend actually reports rather than a constant
// this scenario assumes.
inline iom::WorkspaceRequirements observed_maximum_requirement(
        const DestinationBinding& binding) {
    iom::WorkspaceRequirements maximum;
    for (const std::unique_ptr<iom::Tensor>& owner : binding.owners) {
        const iom::WorkspaceRequirements required =
                owner->view().copy_from_host_workspace_requirements();
        maximum.bytes = std::max(maximum.bytes, required.bytes);
        maximum.alignment = std::max(maximum.alignment, required.alignment);
    }
    return maximum;
}

// Queries the preflight of one complete ordered binding and requires it to be
// exactly the maximum serial per-owner requirement of the same binding, with
// the documented zero-byte `{0, 1}` policy: the result is never a sum of
// mutually exclusive scratch ranges, never an aggregate logical byte count,
// and never a guessed size. A CPU or TTNN binding reports `{0, 1}` here
// because every real owner query of that device reports it.
inline iom::WorkspaceRequirements query_requirement(
        iom::Device& device, const iom::ModelSource& source,
        const DestinationBinding& binding, std::string_view context) {
    const iom::WorkspaceRequirements reported =
            source.upload_workspace_requirements(device, binding.span());
    const iom::WorkspaceRequirements observed =
            observed_maximum_requirement(binding);
    if (observed.bytes == 0) {
        CHECK_MESSAGE(reported == (iom::WorkspaceRequirements{0, 1}),
                      context
                              << ": an all-zero binding must report exactly "
                                 "{0, 1}");
    } else {
        CHECK_MESSAGE(reported == observed,
                      context
                              << ": the preflight must report the maximum "
                                 "serial per-owner requirement of the binding");
    }
    return reported;
}

// Provisions the reusable caller scratch of one reported requirement, and only
// when it is positive: a `{0, 1}` binding consumes no workspace at all and
// therefore never calls a positive raw-workspace factory, which CPU and TTNN
// reject.
inline std::unique_ptr<iom::RawWorkspace> provision_scratch(
        iom::Device& device, const iom::WorkspaceRequirements& reported) {
    if (reported.bytes == 0) {
        return nullptr;
    }
    std::unique_ptr<iom::RawWorkspace> workspace =
            device.create_workspace(reported.bytes);
    REQUIRE(workspace != nullptr);
    return workspace;
}

// The exact scratch view one binding realizes with: a provisioned owner, or the
// valid empty default that a zero-byte requirement consumes.
inline iom::RawWorkspaceView scratch_of(
        const std::unique_ptr<iom::RawWorkspace>& workspace) {
    return workspace == nullptr ? iom::RawWorkspaceView{} : workspace->view();
}

// Requires every destination of one binding to hold exactly the fixture bytes
// of its published index, read back through a real `TensorView` transfer that
// queries the separate download requirement. The expectation is the fixture's
// own entry list, never a value the loader was asked for, and it covers the
// rank-one normalization payloads realized into their `[1, H]` destinations.
inline void require_realized_bytes(
        const DestinationBinding& binding,
        const std::vector<iom_model_loading::SafetensorsEntry>& entries,
        std::string_view context) {
    REQUIRE(binding.owners.size() == entries.size());
    for (std::size_t index = 0; index < entries.size(); ++index) {
        CAPTURE(index);
        const std::string& expected = entries[index].payload;
        const std::vector<std::byte> actual =
                iom_conformance::read_logical(binding.owners[index]->view());
        REQUIRE(actual.size() == expected.size());
        CHECK_MESSAGE(std::memcmp(actual.data(), expected.data(),
                                  actual.size()) == 0,
                      context << ": realized bytes diverge from fixture entry "
                              << entries[index].name);
    }
}

// Realizes one published source on one device and verifies every realized role
// against the fixture's independent bytes.
//
// The documented caller order is exercised exactly: create the destinations
// from the published specifications, query the complete binding, provision
// reusable scratch from the reported maximum only when it is positive, and
// only then upload. A positive requirement refuses the empty default scratch
// before the first copied role and leaves every seeded destination sentinel
// intact, so that validation failure is observably upload-free; a `{0, 1}`
// binding realizes with the default empty view.
inline void realize_and_verify(
        iom::Device& device, const iom::ModelSource& source,
        const std::vector<iom_model_loading::SafetensorsEntry>& entries,
        std::string_view context) {
    DestinationBinding binding = create_binding(device, source);
    REQUIRE(binding.owners.size() == entries.size());

    // Every owner was created from the source's own selected specification.
    for (std::size_t index = 0; index < entries.size(); ++index) {
        CAPTURE(index);
        CHECK(binding.owners[index]->view().spec() ==
              source.tensor_spec(index));
    }

    const iom::WorkspaceRequirements reported =
            query_requirement(device, source, binding, context);

    if (reported.bytes != 0) {
        // A positive requirement is validated before the first copied role, so
        // the empty default scratch is refused and nothing is uploaded.
        const std::vector<std::vector<std::byte>> sentinels =
                seed_sentinels(binding);
        CHECK_THROWS_AS(
                (void)source.upload_weights(device, binding.span()),
                std::invalid_argument);
        require_sentinels(binding, sentinels, context);
    }

    const std::unique_ptr<iom::RawWorkspace> scratch =
            provision_scratch(device, reported);
    source.upload_weights(device, binding.span(), scratch_of(scratch));
    require_realized_bytes(binding, entries, context);
}

// ---------------------------------------------------------------------------
// Scenarios.
// ---------------------------------------------------------------------------

// One complete checkpoint case. The fixture writes `config` and its required
// entries; the published inventory and the selected metadata of every index
// are checked against the fixture's independent expectation; the same
// published source is realized byte-identically on the driver's CPU reference
// device and on the selected device; and the realized weights then outlive the
// deleted artifacts and the released source.
inline void run_checkpoint_case(const ConformanceDevices& devices,
                                const nlohmann::json& config,
                                std::string_view name) {
    const std::vector<iom_model_loading::SafetensorsEntry> entries =
            iom_model_loading::required_weight_entries(config);
    const std::vector<iom_model_loading::ExpectedWeight> expected =
            iom_model_loading::expected_inventory(config);
    REQUIRE(entries.size() == expected.size());

    const iom_model_loading::TempDir dir(fixture_tag(devices, name));
    std::unique_ptr<iom::ModelSource> source =
            load_checkpoint(dir.path(), config, entries);

    CAPTURE(std::string(name));
    CHECK(source->weights().size() == expected.size());
    iom_model_loading::check_inventory(*source, expected);
    CHECK(iom_model_loading::same_config(
            source->config(), iom::load_tinyllama_config(dir.path())));
    // The derived head plan is the configuration's own runtime value.
    CHECK(source->config().head_dim ==
          config.at("hidden_size").get<std::size_t>() /
                  config.at("num_attention_heads").get<std::size_t>());

    const std::string label(fixture_tag(devices, name));

    // The same published source realizes on the independent CPU reference
    // device and on the selected device: there is no backend-specific loader
    // port and no per-backend expectation, and both realize the documented
    // inventory order bit-for-bit.
    realize_and_verify(devices.reference, *source, entries, label + " reference");
    realize_and_verify(devices.candidate, *source, entries, label + " candidate");

    // The realized weights belong to the caller's destination owners alone:
    // delete the fixture artifacts, keep the source alive through the last
    // metadata-borrowing call, then release it and read the same binding again.
    DestinationBinding binding = create_binding(devices.candidate, *source);
    const iom::WorkspaceRequirements reported =
            query_requirement(devices.candidate, *source, binding, label);
    const std::unique_ptr<iom::RawWorkspace> scratch =
            provision_scratch(devices.candidate, reported);
    source->upload_weights(devices.candidate, binding.span(), scratch_of(scratch));
    require_realized_bytes(binding, entries, label + " retained");

    std::error_code error;
    CHECK(std::filesystem::remove(dir.path() / "model.safetensors", error));
    CHECK(std::filesystem::remove(dir.path() / "config.json", error));
    source.reset();
    require_realized_bytes(binding, entries, label + " after release");
}

// Valid extras change no selected result: the same checkpoint plus extra
// tensors of other dtypes and shapes, a second shard of unrelated tensors, and
// unrelated documents publishes the identical inventory and realizes the
// identical required-role bytes.
inline void run_extras_case(const ConformanceDevices& devices,
                            const nlohmann::json& config,
                            std::string_view name) {
    const std::vector<iom_model_loading::SafetensorsEntry> entries =
            iom_model_loading::required_weight_entries(config);
    const std::vector<iom_model_loading::ExpectedWeight> expected =
            iom_model_loading::expected_inventory(config);
    const std::size_t hidden = config.at("hidden_size").get<std::size_t>();

    const iom_model_loading::TempDir dir(fixture_tag(devices, name));
    iom_model_loading::write_config(dir.path(), config);
    std::vector<iom_model_loading::SafetensorsEntry> with_extras = entries;
    with_extras.push_back(iom_model_loading::SafetensorsEntry{
            "model.layers.0.self_attn.rotary_emb.inv_freq", "F32", {2},
            iom_model_loading::deterministic_payload(8, 0x91)});
    with_extras.push_back(iom_model_loading::SafetensorsEntry{
            "extra.u8.table", "U8", {5},
            iom_model_loading::deterministic_payload(5, 0x92)});
    with_extras.push_back(iom_model_loading::bf16_entry(
            "extra.unused.weight", {hidden, hidden}, 0x93));
    iom_model_loading::write_safetensors_file(dir.path(), "model.safetensors",
                                              with_extras);
    iom_model_loading::write_safetensors_file(
            dir.path(), "extra-shard.safetensors",
            {iom_model_loading::bf16_entry("shard.only.extra", {2, 2}, 0x94)});
    iom_model_loading::write_file(dir.path(), "tokenizer.json",
                                  "not json at all");
    iom_model_loading::write_file(dir.path(), "generation_config.json",
                                  "{\"do_sample\":true}");

    const std::unique_ptr<iom::ModelSource> source =
            iom::load_tinyllama_safetensors(dir.path());
    REQUIRE(source != nullptr);

    CAPTURE(std::string(name));
    CHECK(source->weights().size() == expected.size());
    iom_model_loading::check_inventory(*source, expected);
    realize_and_verify(devices.candidate, *source, entries,
                       fixture_tag(devices, name) + " extras");
}

// A rejected binding uploads nothing. Every violation is formed from fresh
// destinations on the selected device, seeded with their own sentinels, and
// must leave every byte of every destination exactly as seeded; the preflight
// refuses the same binding with the same category, so no partial requirement
// is reported either.
inline void run_binding_rejection_case(const ConformanceDevices& devices,
                                       std::string_view name) {
    const nlohmann::json config = iom_model_loading::two_layer_config();
    const std::vector<iom_model_loading::SafetensorsEntry> entries =
            iom_model_loading::required_weight_entries(config);

    const iom_model_loading::TempDir dir(fixture_tag(devices, name));
    const std::unique_ptr<iom::ModelSource> source =
            load_checkpoint(dir.path(), config, entries);

    iom::Device& device = devices.candidate;
    REQUIRE(source->weights().size() == entries.size());
    const std::string label(fixture_tag(devices, name));

    // A valid scratch range satisfies the workspace rule, so each rejection
    // below is a complete-binding violation and never a scratch one. Its owner
    // stays alive for the whole case: a released workspace view would be a dead
    // owner rather than the valid scratch these rejections must be judged with.
    const std::unique_ptr<iom::RawWorkspace> probe_scratch = [&device, &source,
                                                             &label] {
        const DestinationBinding probe = create_binding(device, *source);
        return provision_scratch(
                device,
                query_requirement(device, *source, probe, label + " probe"));
    }();
    const iom::RawWorkspaceView scratch_range = scratch_of(probe_scratch);

    const auto rejected = [&device, &source, &scratch_range](
                                  DestinationBinding& binding,
                                  const char* what) {
        CAPTURE(what);
        const std::vector<std::vector<std::byte>> sentinels =
                seed_sentinels(binding);
        CHECK_THROWS_AS(
                (void)source->upload_workspace_requirements(device,
                                                            binding.span()),
                std::invalid_argument);
        require_sentinels(binding, sentinels, what);
        CHECK_THROWS_AS(
                (void)source->upload_weights(device, binding.span(),
                                             scratch_range),
                std::invalid_argument);
        require_sentinels(binding, sentinels, what);
    };

    // A schema mismatch at the last published index: the final role bound to
    // its transposed orientation, after the rest of the binding was already
    // accepted position by position.
    {
        DestinationBinding binding = create_binding(device, *source);
        std::vector<std::size_t> dims = iom_model_loading::dims_of(
                source->tensor_spec(binding.owners.size() - 1).shape);
        REQUIRE(dims.size() == 2);
        std::swap(dims[0], dims[1]);
        binding.owners.back() = device.create_tensor(iom::TensorSpec{
                iom::TensorShape{std::move(dims)}, iom::DataType::BF16,
                iom::QuantizationFormat::NONE});
        binding.pointers.back() = binding.owners.back().get();
        rejected(binding, "transposed final specification");
    }

    // A destination created by another Device instance: the driver-owned
    // `foreign` device is a distinct Device, and for a backend that can only
    // host one instance in process it is a distinct device of another kind, so
    // exact Device identity and never "an equal backend and ordinal" is what
    // the binding must enforce.
    {
        DestinationBinding binding = create_binding(device, *source);
        CHECK(&devices.foreign != &device);
        binding.owners.back() = devices.foreign.create_tensor(
                source->tensor_spec(binding.owners.size() - 1));
        binding.pointers.back() = binding.owners.back().get();
        rejected(binding, "foreign device destination");
    }

    // The last published index unbound, one destination too few, one too many,
    // and one owner bound to two roles whose selected metadata is equal: the
    // two `[1, H]` normalization scales of the first decoder layer.
    {
        DestinationBinding binding = create_binding(device, *source);
        binding.pointers.back() = nullptr;
        rejected(binding, "null final destination");
    }
    {
        DestinationBinding binding = create_binding(device, *source);
        binding.pointers.pop_back();
        rejected(binding, "one destination too few");
    }
    {
        DestinationBinding binding = create_binding(device, *source);
        const std::unique_ptr<iom::Tensor> excess =
                device.create_tensor(source->tensor_spec(0));
        binding.pointers.push_back(excess.get());
        rejected(binding, "one destination too many");
    }
    {
        DestinationBinding binding = create_binding(device, *source);
        REQUIRE(source->tensor_spec(3) == source->tensor_spec(1));
        binding.pointers[3] = binding.pointers[1];
        rejected(binding, "repeated owner");
    }
}

// A checkpoint whose final required role is absent, and one whose final role
// carries the wrong orientation, are model-schema rejections: no source is
// published, so no binding can exist and no upload can happen.
inline void run_final_role_rejection_case(const ConformanceDevices& devices,
                                          std::string_view name) {
    const nlohmann::json config = iom_model_loading::two_layer_config();
    const std::vector<iom_model_loading::SafetensorsEntry> entries =
            iom_model_loading::required_weight_entries(config);
    const std::size_t layers = config.at("num_hidden_layers").get<std::size_t>();
    const std::size_t hidden = config.at("hidden_size").get<std::size_t>();
    const std::size_t intermediate =
            config.at("intermediate_size").get<std::size_t>();
    const std::string final_name = "model.layers." +
                                   std::to_string(layers - 1) +
                                   ".mlp.down_proj.weight";
    const std::string required_shape =
            "[" + std::to_string(hidden) + ", " +
            std::to_string(intermediate) + "]";
    const std::string actual_shape =
            "[" + std::to_string(intermediate) + ", " +
            std::to_string(hidden) + "]";
    CAPTURE(final_name);

    // The final role's name is absent while the store size stays complete, so
    // this is the required-name schema check rather than the incomplete-store
    // guard.
    {
        const iom_model_loading::TempDir dir(
                fixture_tag(devices, std::string(name) + "-missing-final-role"));
        std::vector<iom_model_loading::SafetensorsEntry> variant = entries;
        const auto removed = std::remove_if(
                variant.begin(), variant.end(),
                [&final_name](
                        const iom_model_loading::SafetensorsEntry& entry) {
                    return entry.name == final_name;
                });
        REQUIRE(removed != variant.end());
        variant.erase(removed, variant.end());
        variant.push_back(iom_model_loading::bf16_entry(
                final_name + ".unused", {hidden, intermediate}, 0xA1));

        bool rejected = false;
        try {
            (void)load_checkpoint(dir.path(), config, variant);
        } catch (const std::invalid_argument& error) {
            rejected = true;
            const std::string message = error.what();
            CHECK(message.find(dir.path().string()) != std::string::npos);
            CHECK(message.find(final_name) != std::string::npos);
            CHECK(message.find("<missing>") != std::string::npos);
        }
        CHECK_MESSAGE(rejected, "a missing final required role must be a "
                                "model-schema rejection");
    }

    // The final role is present with the transposed orientation, so the
    // required rank/shape check and not the missing-name check rejects it.
    {
        const iom_model_loading::TempDir dir(
                fixture_tag(devices, std::string(name) + "-final-schema"));
        std::vector<iom_model_loading::SafetensorsEntry> variant = entries;
        bool replaced = false;
        for (iom_model_loading::SafetensorsEntry& entry : variant) {
            if (entry.name == final_name) {
                entry = iom_model_loading::bf16_entry(
                        final_name, {intermediate, hidden}, 0xA2);
                replaced = true;
            }
        }
        REQUIRE(replaced);

        bool rejected = false;
        try {
            (void)load_checkpoint(dir.path(), config, variant);
        } catch (const std::invalid_argument& error) {
            rejected = true;
            const std::string message = error.what();
            CHECK(message.find(final_name) != std::string::npos);
            CHECK(message.find(required_shape) != std::string::npos);
            CHECK(message.find(actual_shape) != std::string::npos);
        }
        CHECK_MESSAGE(rejected, "a wrong final role schema must be a "
                                "model-schema rejection");
    }
}

}  // namespace model_loading

// The one shared model-loading scenario every backend driver calls with its own
// devices: the same synthetic checkpoints are loaded, realized, and verified
// on the CPU reference device and on the selected device of the driver.
//
// `candidate` is the device under test and `foreign` is the driver-owned
// second instance of the same backend used to reject a foreign destination;
// only the driver creates those devices and their allocators.
inline void run_model_loading_conformance(const ConformanceDevices& devices) {
    // The one-layer and two-layer boundary checkpoints, then the H16 and H18
    // normalization/tile-boundary checkpoints.
    model_loading::run_checkpoint_case(devices, iom_model_loading::two_layer_config(),
                                       "two-layer");
    model_loading::run_checkpoint_case(devices, iom_model_loading::one_layer_config(),
                                       "one-layer");
    model_loading::run_checkpoint_case(devices, iom_model_loading::h16_config(),
                                       "h16");
    model_loading::run_checkpoint_case(devices, iom_model_loading::h18_config(),
                                       "h18");

    // Valid extras, then every rejected binding, then the rejected final role.
    model_loading::run_extras_case(devices, iom_model_loading::two_layer_config(),
                                   "two-layer-extras");
    model_loading::run_binding_rejection_case(devices, "binding-rejection");
    model_loading::run_final_role_rejection_case(devices, "final-role");
}

}  // namespace iom_conformance