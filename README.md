# xpu-serve
AI written experimental engine to run LLMs on Intel platforms.

One repo, one binary: the model code, the engine, the tokenizer, chat templates and an OpenAI-compatible server all
live here, and the math runs on an Intel GPU through a backend (OpenVINO first, SYCL next). Read
[DESIGN.md](DESIGN.md) for the why and [TODO.md](TODO.md) for the plan.

**Status: v0.3.** Gemma 4 E2B / E4B (text and images) and Qwen3.5 (dense, text), weights loaded straight from the
Hugging Face checkpoint (no convert step yet), one request at a time, OpenVINO backend. [PROJECT.md](PROJECT.md) maps
the code.

## Build

Needs a C++20 compiler (g++ 11 or newer), CMake 3.22+, Ninja, and the OpenVINO 2026.2 runtime archive.

```bash
source /opt/intel/openvino_2026.2.1/setupvars.sh        # or wherever OpenVINO is unpacked
cmake -S . -B build -G Ninja -DOpenVINO_DIR=$INTEL_OPENVINO_DIR/runtime/cmake
cmake --build build
build/tests/xs_tests                                     # unit tests
```

## Run

Any Hugging Face directory of a supported model works, e.g. `google/gemma-4-E2B-it` or `Qwen/Qwen3.5-4B`:

```bash
build/src/xpu-serve info                                           # backends and devices
build/src/xpu-serve run   ~/models/gemma-4-E2B-it --device GPU     # chat in the terminal
build/src/xpu-serve run   ~/models/gemma-4-E2B-it --device GPU --image photo.jpg --prompt "What is in this picture?"
build/src/xpu-serve serve ~/models/gemma-4-E2B-it --device GPU     # http://127.0.0.1:8080/v1
build/src/xpu-serve bench ~/models/gemma-4-E2B-it --device GPU --prompt-tokens 1024 --gen-tokens 128
```

The server speaks the OpenAI API (`/v1/chat/completions` with streaming, tool calls and images, `/v1/completions`,
`/v1/models`), so any OpenAI client can point at it. Images go in as `image_url` parts with a base64 `data:` URL
(PNG, JPEG, BMP or GIF); the server does not fetch remote URLs.

## Correctness

Every layer is tested against a reference:

| What | Reference | Where |
|---|---|---|
| Tokenizer (Gemma BPE, byte-level BPE with NFC) | Hugging Face `tokenizers`, exact ids and text | `tests/unit/tokenizer/` |
| Chat template | `apply_chat_template`, byte-exact | `tests/unit/chat/` |
| Image resize | torchvision's antialiased bicubic, bit-exact | `tests/unit/image/` |
| Every graph op, both backends | fp64 references, with planted-bug controls | `tests/unit/backend/` |
| Whole model (Gemma 4, Qwen3.5) | Hugging Face fp32 logits and greedy output | `tests/unit/model/model_parity.cpp` |
| Image input | Hugging Face fp32 pixels, vision features, logits and greedy output | `tests/unit/model/gemma4_vision_parity.cpp` |

The model parity test needs the model and generated fixtures:

```bash
python -I tests/parity/model/hf_reference.py --model-dir ~/models/gemma-4-E2B-it          # needs transformers
python -I tests/parity/model/hf_vision_reference.py --model-dir ~/models/gemma-4-E2B-it   # image cases
XS_TEST_MODEL_DIR=~/models/gemma-4-E2B-it XS_PARITY_BACKEND=openvino build/tests/xs_tests -tc='gemma4*'
```
