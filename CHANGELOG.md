# Changelog

## v1.1.0

- Apple-Silicon (MLX/MPS) support across CLI features.
- Device-aware runtime with unified detection and banners.
- Optional CUDA-only extras guarded with platform markers.
- Portable quantization via MLX (q4/q8) and Optimum-Quanto (int4/int8).
- Device-agnostic sparsity/pruning with 2:4 pattern.
- Portable inference engine with streaming.
- Profiling without NVML assumptions; RSS-based memory stats.
- FastAPI serving backends: MLX and llama.cpp; vLLM gated.
- Tests, smoke script, and macOS CI workflow.
