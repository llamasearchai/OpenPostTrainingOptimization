# OpenPostTrainingOptimizations Architecture

## Overview

OpenPostTrainingOptimizations is a portable post-training optimization toolkit designed to run on Apple Silicon (MLX/MPS/CPU) and Linux/CUDA platforms. The platform integrates advanced AI capabilities including OpenAI Agents SDK, Ollama, LLM toolkit, and Datasette for comprehensive model optimization and deployment workflows.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                          │
├────────────────────────┬────────────────────────────────────────────┤
│    CLI (opt)           │        TUI (opt-tui)                        │
└────────────────────────┴────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                     Core Optimization Layer                          │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────┤
│ Quantization │   Sparsity   │  Speculative │  Profiling   │ Serving │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                      AI Integration Layer                            │
├──────────────────┬──────────────────┬───────────────────────────────┤
│  OpenAI Agents   │     Ollama       │   LLM Toolkit                 │
└──────────────────┴──────────────────┴───────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Management Layer                           │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   Datasette      │  sqlite-utils    │   SQLite Backend              │
└──────────────────┴──────────────────┴───────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                      Hardware Abstraction                            │
├────────────┬────────────┬────────────┬────────────┬─────────────────┤
│    MLX     │    MPS     │    CUDA    │    CPU     │  Triton Kernels │
└────────────┴────────────┴────────────┴────────────┴─────────────────┘
```

## Module Breakdown

### 1. Core Optimization Modules

#### Quantization (`core/quantization.py`)
- Supports INT8, INT4, GPTQ, AWQ quantization methods
- Backend-specific implementations (MLX, MPS/Quanto, CUDA/bitsandbytes)
- Automatic size estimation and benchmarking
- Persistent configuration and metadata storage

#### Sparsity (`core/sparsity.py`)
- Unstructured and structured sparsity (N:M patterns)
- Magnitude-based pruning
- Sparsity metrics and theoretical speedup calculation

#### Speculative Decoding (`core/speculative.py`)
- Draft-verify architecture for accelerated generation
- Configurable gamma parameters
- Acceptance rate tracking and benchmarking

### 2. AI Agents Integration

#### OpenAI Agents SDK (`agents/`)
- Tool/function calling with predefined optimization tools
- Streaming chat support
- Async operation support
- Safe API key management via environment variables
- Conversation history tracking

#### Ollama Integration (`integrations/ollama_integration.py`)
- Model lifecycle management (pull, list, delete)
- Chat and generation interfaces
- Embeddings generation
- Model information retrieval

#### LLM Toolkit (`integrations/llm_integration.py`)
- CLI wrapper for Simon Willison's llm toolkit
- Plugin management (llm-ollama)
- Prompt logging and history
- llm-cmd integration for command generation

### 3. Data Management

#### Datasette Integration (`integrations/datasette_integration.py`)
- SQLite database schema:
  - `prompts`: AI agent prompt/response pairs
  - `runs`: Optimization job tracking
  - `traces`: Operation-level traces
  - `datasets`: Dataset metadata
  - `metrics`: Performance metrics
  - `models`: Model metadata
- Query interface
- Data export (JSON, CSV)
- Automatic database initialization

### 4. Terminal User Interface

#### TUI Module (`tui/`)
- Built with Textual framework
- Multiple screens:
  - Welcome/Main Menu
  - Quantization
  - Sparsity
  - Profiling
  - Serving
  - AI Agents
  - Datasette Browser
  - Settings
- Keyboard navigation and shortcuts
- Config persistence
- Theme support (dark/light)

### 5. Model Serving

#### Inference Engine (`deployment/inference_engine.py`)
- Unified interface for model loading and generation
- Backend-agnostic design
- Streaming generation support
- Temperature and sampling controls

#### Server Implementations (`deployment/servers/`)
- MLX Server: FastAPI-based serving for Apple Silicon
- llama.cpp Server: GGUF model serving with Metal acceleration
- Extensible server architecture

## Data Flow

### Optimization Workflow

```
User Request (CLI/TUI)
    ↓
Device Detection (hardware_utils)
    ↓
Model Loading
    ↓
Optimization Operation (quantize/sparsify/profile)
    ↓
Metrics Collection
    ↓
Results Storage (Datasette)
    ↓
Output/Export
```

### AI Agent Workflow

```
User Prompt
    ↓
Agent Runner (OpenAI/Ollama)
    ↓
Tool Execution (if needed)
    ↓
Response Generation
    ↓
Conversation History Update
    ↓
Prompt/Response Storage (Datasette)
```

## Technology Stack

### Core Dependencies
- Python 3.10+
- PyTorch (MPS/CUDA support)
- Transformers (Hugging Face)
- Rich (Terminal UI rendering)

### Optimization Libraries
- MLX (Apple Silicon)
- optimum-quanto (INT8/INT4 quantization)
- bitsandbytes (CUDA quantization)
- Triton (Custom CUDA kernels)

### AI Integration
- openai (OpenAI Agents SDK)
- ollama (Ollama client)
- llm (Simon Willison's toolkit)

### Data Management
- datasette (Data browsing)
- sqlite-utils (SQLite operations)
- SQLAlchemy (Database abstraction)

### UI Frameworks
- Textual (TUI)
- FastAPI (Web serving)
- Typer (CLI parsing)

## Deployment Architecture

### Docker Deployment

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose Stack                   │
├───────────────┬──────────────────┬──────────────────────┤
│  App Container│ Ollama Container │ Datasette Container  │
│  (port 8000)  │  (port 11434)    │  (port 9000)         │
└───────────────┴──────────────────┴──────────────────────┘
         │               │                    │
         └───────────────┴────────────────────┘
                         │
                   Bridge Network
```

### Multi-stage Build
1. Builder stage: Dependency installation and package building
2. Runtime stage: Slim image with application code
3. Development stage: Extended with dev tools

## Security Considerations

- Non-root container execution
- No hardcoded secrets (environment variables only)
- Secret scanning (gitleaks, bandit)
- Container vulnerability scanning (Trivy)
- SBOM generation for supply chain security
- CORS configuration for API endpoints

## Performance Optimizations

- Lazy imports for optional dependencies
- Backend-specific kernel selection
- Device synchronization for accurate benchmarking
- Streaming generation for reduced latency
- Multi-stage Docker builds with caching
- Parallel test execution with pytest-xdist

## Extensibility

### Adding New Quantization Methods
1. Implement in `core/quantization.py`
2. Add backend detection logic
3. Update tool definitions in `agents/tools.py`
4. Add tests in `tests/`

### Adding New Backends
1. Extend `hardware_utils.py` detection
2. Implement backend-specific loaders
3. Add device synchronization support
4. Update CI/CD for backend-specific tests

### Adding New AI Tools
1. Define tool schema in `agents/tools.py`
2. Implement tool function
3. Update OpenAI function definitions
4. Add tool to agent runner

## Testing Strategy

- Unit tests for all modules
- Integration tests for cross-module workflows
- E2E tests for CLI/TUI flows
- Property-based tests (hypothesis)
- Performance benchmarks
- Security tests (bandit, trivy)
- Multi-platform CI (Linux, macOS)
- Multi-Python version testing (3.10, 3.11, 3.12)

## Monitoring and Observability

- Structured logging with levels
- Performance metrics collection
- Database-backed run tracking
- Healthcheck endpoints
- Coverage reporting (Codecov)
- SBOM generation for dependency tracking

## Author

Nik Jois

