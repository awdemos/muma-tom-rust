# MuMA-ToM Rust Implementation

A production-ready Rust implementation of the MuMA-ToM (Multi-modal Multi-Agent Theory of Mind) benchmark and LIMP (Language model-based Inverse Multi-agent Planning) model, based on [arXiv:2408.12574v1](https://arxiv.org/abs/2408.12574v1).

## Overview

This project implements:
- **MuMA-ToM Benchmark**: 225 multi-modal social interactions with 900 multi-choice questions
- **LIMP Model**: Multi-modal fusion + hypothesis parsing + Bayesian inverse planning
- **Integration**: GPT-4o (OpenAI) and Gemini 1.5 Pro (Google)
- **Features**: Video processing, multi-modal fusion, cost optimization with caching

## Architecture

```
src/
├── api_clients/      # LLM/VLM API integrations
│   ├── openai.rs     # GPT-4o client
│   ├── gemini.rs     # Gemini 1.5 Pro client
│   └── unified.rs    # Multi-provider gateway
├── benchmark/         # Benchmark runner and data loading
├── fusion/            # Multi-modal fusion module
├── hypothesis.rs      # Hypothesis parsing
├── inverse_planning.rs # Bayesian inverse planning
├── models.rs          # Core data models
├── config.rs          # Configuration management
├── error.rs           # Error handling
└── utils.rs           # Utilities
```

## Quick Start

```bash
# Install dependencies
cargo build

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run benchmark
cargo run -- --benchmark data/benchmark

# Run tests
cargo test
```

## Configuration

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=AIza-your-gemini-key-here
```

## Benchmark Structure

The MuMA-ToM benchmark includes:
- **225 interactions**: Multi-agent social scenarios in household environments
- **900 questions**: 3 question types (300 each)
  1. **Belief Inference**: True/false beliefs about physical states
  2. **Social Goal Inference**: Helping, hindering, independent actions
  3. **Belief of Goal Inference**: What one agent believes about another's goal

## License

CC BY 4.0 - See LICENSE file for details

## References

- Paper: [MuMA-ToM: Multi-modal Multi-Agent Theory of Mind](https://arxiv.org/abs/2408.12574v1)
- Original Implementation: https://github.com/SCAI-JHU/MuMA-ToM
