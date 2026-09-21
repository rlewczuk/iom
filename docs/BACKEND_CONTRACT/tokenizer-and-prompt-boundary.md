# 11. Tokenizer and prompt boundary

Tokenizer and chat formatting are backend-neutral host contracts. They consume
only caller-selected model-directory artifacts and produce host token IDs or
host prompt bytes; they do not select a device, allocate a tensor, submit
queued work, or depend on a backend runtime. This section is the sole
normative owner of the tokenizer/formatter boundary and its composition. A
backend MUST consume the resulting IDs and bytes without changing them.

## Public owners and APIs

`Tokenizer` and `ChatFormatter` are final, immutable, non-copyable, and
non-movable private-pImpl owners. Their factories return
`std::unique_ptr<Tokenizer>` and `std::unique_ptr<ChatFormatter>` only after
the complete artifact, schema, and compilation validation succeeds. They retain
their private state for their whole owner lifetime and publish no partially
built state. Owning host strings and vectors are constructed privately before a
successful return; an exception never exposes a partial result. A
`ChatMessageView` borrows the caller's role and content strings for the
duration of the `format` call and is never retained.

The public declarations are:

```cpp
struct EncodeOptions {
    bool add_special_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
};

class Tokenizer final {
public:
    std::vector<std::uint32_t> encode(
        std::string_view text, EncodeOptions options) const;
    std::vector<std::uint32_t> encode_pair(
        std::string_view first, std::string_view second,
        EncodeOptions options) const;
    std::string decode(
        std::span<const std::uint32_t> ids, DecodeOptions options) const;
    std::uint32_t bos_id() const;
    std::uint32_t eos_id() const;
    std::uint32_t unk_id() const;
    std::uint32_t tokenizer_pad_id() const;
};

std::unique_ptr<Tokenizer> load_tokenizer(
    const std::filesystem::path& model_directory);

struct ChatMessageView {
    std::string_view role;
    std::string_view content;
};

class ChatFormatter final {
public:
    std::string format(
        std::span<const ChatMessageView> messages,
        bool add_generation_prompt) const;
};

std::unique_ptr<ChatFormatter> load_chat_formatter(
    const std::filesystem::path& model_directory,
    std::optional<std::string_view> template_override = std::nullopt);
```

The tokenizer IDs are fixed as `UNK=0`, `BOS=1`, `EOS=2`, and
`tokenizer_pad_id()=2`. The generation/batching pad policy is separate:
`generation_config.json` records generation pad ID `0`; that value MUST NOT be
treated as the tokenizer pad ID or silently conflated with tokenizer padding.

## Authoritative artifacts and tokenizer semantics

The caller supplies one explicit `--model-dir`; every artifact below is
resolved relative to that directory. `tokenizer.json` is the sole runtime
tokenizer authority. Its pinned evidence anchors are
`tokenizer.json:1-33,34-50,51-104,105-128,129-137`, covering its
added-token table, normalizer and absent pre-tokenizer, TemplateProcessing
post-processor, decoder, and BPE model configuration. `tokenizer.model` is a
cross-check-only SentencePiece artifact: the runtime tokenizer MUST NOT read
it, substitute it, or require it.
The accepted `tokenizer.json` schema is pinned, not a generic tokenizer
loader: `version` is `1.0`, `truncation` and `padding` are `null`,
`added_tokens` contains exactly IDs `0`, `1`, and `2` with the contents and
flags above, `normalizer` is the stated two-step sequence,
`pre_tokenizer` is `null`, `post_processor` is TemplateProcessing, `decoder`
is the stated sequence, and `model` is the BPE configuration with
`dropout=null`, no continuing-subword prefix or end-of-word suffix,
`unk_token="<unk>"`, `fuse_unk=true`, and `byte_fallback=true`. The published
vocabulary and ordered merges are consumed as artifact data; malformed,
missing, duplicate, or unsupported schema fields fail validation. The
tokenizer factory reads this file only and publishes no tokenizer on failure.

`encode` accepts valid UTF-8 only. The added tokens `<unk>`, `<s>`, and `</s>`
are recognized as exact atomic tokens before normalization, even when
`add_special_tokens` is `false`; they are not normalized, split, or
heuristically deduplicated. The normalizer first prepends U+2581 (`▁`) and
then replaces every ASCII space with U+2581. There is no pre-tokenizer.
Unicode pieces and byte-fallback pieces are selected by the pinned BPE model,
unknown spans use fused UNK behavior, and the BPE merge list is applied in its
stored order.

When `add_special_tokens` is `true`, the TemplateProcessing post-processor
adds BOS ID `1` to a single sequence (`[BOS] A`) and adds BOS ID `1` before
each sequence of a pair (`[BOS] A [BOS] B`). It adds no EOS. It performs no
heuristic duplicate removal: an explicit `<s>` in the input remains an
explicit ID in addition to any processor-added BOS. When
`add_special_tokens` is `false`, processor insertion is suppressed but
explicit added-token recognition remains active. Raw role markers such as
`<|user|>`, `<|system|>`, and `<|assistant|>` are ordinary text to this
tokenizer, not special tokens.

`decode` validates every ID before producing output. With
`skip_special_tokens=false` it retains all IDs, including `0`, `1`, and `2`;
with `skip_special_tokens=true` it removes only those three IDs. Its decoder
sequence is exactly Replace U+2581 with an ASCII space, ByteFallback, Fuse,
and `Strip(space,start=1,stop=0)`. There is no cleanup, Unicode
normalization, role-marker handling, or implicit special-token policy. The
returned `std::string` contains the decoded bytes exactly as produced by that
sequence.

The pinned companion artifacts are evidence and cross-check inputs, not
alternative runtime authorities: `tokenizer_config.json:2-39` owns formatter
metadata, `special_tokens_map.json:2-27` cross-checks the four named tokens,
and `generation_config.json:2-6` records generation policy only. The
formatter never reads generation configuration, and the tokenizer never
switches to `tokenizer.model`.

## Formatter metadata, override, and fixed grammar

`load_chat_formatter` reads only `tokenizer_config.json`. The supported keys
are exactly `added_tokens_decoder`, `bos_token`, `chat_template`,
`clean_up_tokenization_spaces`, `eos_token`, `legacy`, `model_max_length`,
`pad_token`, `padding_side`, `sp_model_kwargs`, `tokenizer_class`,
`unk_token`, and `use_default_system_prompt`; an unsupported key is a schema
error. It validates the complete added-token decoder entries and flags for IDs
`0`, `1`, and `2`:

| ID | content | single_word | lstrip | rstrip | normalized | special |
| --- | --- | --- | --- | --- | --- | --- |
| `0` | `<unk>` | `false` | `false` | `false` | `false` | `true` |
| `1` | `<s>` | `false` | `false` | `false` | `false` | `true` |
| `2` | `</s>` | `false` | `false` | `false` | `false` | `true` |

It also requires `bos_token="<s>"`, `eos_token="</s>"`,
`pad_token="</s>"`, `unk_token="<unk>"`,
`clean_up_tokenization_spaces=false`, `legacy=false`,
`model_max_length=2048`, `padding_side="right"`,
`sp_model_kwargs={}`, `tokenizer_class="LlamaTokenizer"`, and
`use_default_system_prompt=false`. The formatter does not ask `Tokenizer` for
IDs and does not infer or repair metadata from another file.

An engaged `template_override` is authoritative even when it is empty or
invalid. The formatter compiles only that selected source and MUST propagate
its failure; it MUST NOT fall back to `chat_template` after an engaged
override fails. With no engaged override, the configured `chat_template` is
the selected source.

The selected source is a fixed restricted Jinja subset, not general template
syntax. It contains exactly one `for message in messages` loop and one
role switch consisting of one `if` followed by exactly two `elif` branches,
with one branch each for the exact lowercase roles `system`, `user`, and
`assistant`. It may contain one final
`loop.last and add_generation_prompt` condition. Every block has a matched
closure. Each role branch has one content reference followed later by
`eos_token`; the generation branch contains literals only. Both single- and
double-quoted role/content indexes are accepted (`message['role']`,
`message["role"]`, `message['content']`, and `message["content"]`).

Only the escapes `\\`, `\'`, `\"`, `\n`, `\r`, and `\t` are valid. `else`,
nested blocks, comments, filters, calls, alternate variables, arbitrary
expressions, and whitespace-control delimiters are rejected. The source MUST
be UTF-8 with LF line endings; a raw CR is rejected. No non-whitespace literal
text may occur between blocks. The parser reports template byte offsets for
violations rather than accepting general Jinja behavior.

## Compilation, rendering, and composition

Compilation applies exactly the template lowering rules: remove one LF
immediately after a `%}` block close, remove line-leading spaces and tabs
immediately before a `{%` block opener, and preserve every other literal byte,
including the LF after a `}}` expression. These transformations apply only
outside string literals; there is no global trim or whitespace normalization.
The formatter compiles once into immutable three-role affixes, generation
bytes, and the iteration tail. `format` never reparses or mutates that plan.

Before publishing any output, `format` validates every message's exact
lowercase role and valid UTF-8 content. Empty content is valid and remains
empty. An empty message list returns `""` even when
`add_generation_prompt=true`. For nonempty input it preserves content bytes
exactly and emits the compiled role literals, `eos_token`, terminal LFs, and
the generation-prefix LF exactly. It performs no system-message injection,
escaping, cleanup, BOS insertion, or mutable per-call state.
For the pinned template, each role affix is respectively
`<|system|>\n`, `<|user|>\n`, or `<|assistant|>\n`, followed by the
unchanged content, `</s>`, and the compiled iteration-tail LF. The optional
generation bytes are exactly `<|assistant|>\n`.

For the pinned default template, one user message whose content is `x`
renders these exact bytes:

```text
add_generation_prompt=false: "<|user|>\nx</s>\n"
add_generation_prompt=true:  "<|user|>\nx</s>\n<|assistant|>\n"
```

The first string includes its terminal LF; the second includes both the
terminal message LF and the generation-prefix LF. The formatter does not add
BOS. A structured prompt is composed by calling `format` first and then
passing the returned bytes to `Tokenizer::encode`; a raw text prompt calls
`Tokenizer::encode` directly. The tokenizer's processor-BOS policy therefore
still applies to the latter encode call unless the caller disables it, while
an explicit `<s>` in raw text remains an additional atomic token and is never
deduplicated. Decode retains IDs `0/1/2` by default and removes only those
IDs when `skip_special_tokens=true`.

Task 10 owns the sole cross-component composition join: its future focused
case passes formatter output through the tokenizer and checks the resulting
boundary without creating a second formatter/tokenizer contract.

## Errors, atomic publication, references, and focused ownership

Malformed JSON, schema, template, UTF-8 input, token ID, and message role are
contextual `std::invalid_argument` failures. File and other I/O failures are
contextual `std::runtime_error` failures. Checked size arithmetic failures
are `std::overflow_error`; allocation failures propagate as allocation
failures. Every validation step completes before a token vector, decoded
string, compiled formatter, or rendered prompt is published. Diagnostics name
the artifact path and field, or the template byte offset, and include the
expected and actual values where applicable.

The independent offline reference runs with the CPython version in use for
this snapshot (CPython 3.14.4) and the pinned packages
`transformers==4.35.0`, `tokenizers==0.14.1`,
`sentencepiece==0.1.99`, and `jinja2==3.1.2`. The caller supplies an
artifact-relative `--model-dir`; the reference performs no downloads and
contains no hardcoded model path. `test/reference/tokenizer_reference.py`
defines the independent behavior, `tokenizer_reference_manifest.json` records
SHA-256 and byte-size identities for the artifacts,
`generate_tokenizer_oracles.py` creates deterministic outputs, and
`tokenizer_oracles.json` stores those pinned oracle results. Production code
MUST NOT serve as its own tokenizer or formatter oracle.

This contract is backend-neutral and has no backend-specific tokenizer,
formatter, device, or queue variation. Future focused ownership is split
between [`test/test_tokenizer.cpp`](../../test/test_tokenizer.cpp) for artifact,
encode/decode, special-token, pair, error, and atomic-output cases and
[`test/test_chat_format.cpp`](../../test/test_chat_format.cpp) for metadata,
override, grammar, lowering, rendering, error, and empty-list cases. The
task-10 composition join is linked from
[`test/test_chat_format.cpp#composition-join`](../../test/test_chat_format.cpp#composition-join).
These links name future cases only; they do not claim that the tests, builds,
or any backend conformance gate have run.
