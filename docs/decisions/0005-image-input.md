# 0005: Image input (Gemma 4 vision)

Date: 2026-11-07 · Status: accepted

Images enter through the OpenAI `image_url` content part and reach the model as rows that replace token embeddings.

**Encoder as its own graph.** A vision model (`VisionEncoder`, next to the text model in `src/models/<arch>/`) builds a
second, stateless step graph that runs once per image; a graph without KV slots is stateless on every backend. Its
output rows replace the embeddings of the image's soft tokens: the text graph takes two extra inputs, `media`
[T, hidden] and `media_keep` [T, 1], and computes `embed * keep + media`. Model code stays backend-neutral and the
backends gained only general ops: Clamp, Transpose, axial (multi-section) Rope and non-causal Attention.

**One image per run, no padding.** Hugging Face pads every image to the 2520-patch budget and masks the padding out
(as attention keys and in the pooling), which leaves the real patches' results exactly as an unpadded run computes
them. So the tower runs with T = the image's own patch count, and the pooling is a host-built [280, T] matrix applied
with a matmul. On the Arc Pro B60 the first run of the tower costs about 1.8 s of kernel setup (paid at load with a
gray image), a new patch count about 0.1 s once, and later images about 40 ms; padding to the budget, which needs a
key mask, stays the fallback should a plugin rebuild per shape.

**Preprocessing on the host, bit-exact.** Decoding uses stb_image (PNG, JPEG, BMP, GIF; alpha dropped as Pillow's
`convert("RGB")` does). The resize reproduces PyTorch's uint8 antialiased bicubic kernel, which Hugging Face's
torchvision image processor runs, so PNG input reaches the tower with the reference's exact pixel values
(`tests/unit/image/`). JPEG decoders differ from libjpeg by a level here and there.

**f16 safety.** The pooler scales by sqrt(hidden) right before a weightless RMSNorm; that scale can pass the f16
range, and `rms_norm(c x, eps) == rms_norm(x, eps / c^2)`, so the graph folds it into epsilon.

**Prefix cache.** Engine positions are keyed by token id, or for a soft token by (image content hash, row), so a new
image never reuses KV computed for another one. Encoder outputs of the last 16 images are cached by hash.

**Requests.** Only `data:image/...;base64,` URLs are accepted; the server never fetches remote URLs. Each image part
the template renders (`<|image|>` for Gemma 4) must pair with one image; anything else is a 400 naming the part.
Audio and video parts are refused by name. A checkpoint whose vision tower is not supported (Gemma 4 26B/31B
`standardize`) loads text-only with a warning, and image requests to it are refused.
