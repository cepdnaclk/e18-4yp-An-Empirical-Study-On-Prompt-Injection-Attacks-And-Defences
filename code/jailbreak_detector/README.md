# Jailbreak Detector

A tool for detecting jailbreak patterns in LLM responses using another LLM as a judge.

## Overview

This project provides tools to analyze responses from Large Language Models (LLMs) and determine if they show signs of successful jailbreaking. The detector uses another LLM as a judge to evaluate whether responses indicate that a model has been compromised or manipulated to bypass safety measures.

Key features:
- Analyze LLM responses for jailbreak patterns
- Compare model judgments with human evaluations
- Optimize inference for GPU efficiency
- Benchmark multiple models
- Analyze custom datasets

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Analysis

The project is specifically designed to analyze the `output_cleaned_text.csv` dataset containing LLM responses and human evaluations. Use `analyze_dataset.py` to evaluate models against this dataset:

```bash
python analyze_dataset.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --optimize
```

### Dataset Analysis Options

- `--data`, `-d`: Path to dataset CSV file [default: data/llm-as-a-judge/output_cleaned_text.csv]
- `--output-dir`, `-o`: Directory to save results [default: analysis_results]
- `--model`, `-m`: HuggingFace model to use [default: TinyLlama/TinyLlama-1.1B-Chat-v1.0]
- `--sample-size`, `-s`: Number of samples to analyze [all if not specified]
- `--batch-size`, `-b`: Batch size for inference [default: 8]
- `--no-quantize`, `-nq`: Disable 4-bit quantization
- `--device`: Device for inference (cuda, cpu, mps) [default: auto-detected]
- `--max-new-tokens`: Maximum tokens to generate [default: 128] 
- `--temperature`: Generation temperature [default: 0.1]
- `--save-disagreements`: Save cases where model and human evaluations disagree
- `--models-list`, `-ml`: Path to JSON file with models to evaluate
- `--optimize`: Apply all inference optimizations
- `--log-file`: Path to log file

### Multiple Model Evaluation

To evaluate multiple models, create a JSON file with model names:

```bash
python analyze_dataset.py --models-list models_list.json --optimize
```

## GPU Optimization

The tool includes several optimizations for efficient GPU usage:

- 4-bit quantization (reduces memory usage by ~75%)
- Batch processing (processes multiple responses at once)
- Flash Attention 2 support (for Llama models)
- BetterTransformer optimizations
- PyTorch 2.0 compile for faster inference
- Memory-efficient loading
- KV cache for faster generation

## Benchmarking

Use the benchmark script to compare different models:

```bash
python benchmark.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 meta-llama/Llama-2-7b-chat-hf
```

## Evaluating on Custom Data

To analyze your own dataset:

```bash
python main.py --input your_responses.json --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Metrics and Evaluation

The tool evaluates models against human judgments using:

- Accuracy: Overall correctness of predictions
- Precision: Ratio of true positives to all positive predictions
- Recall: Ratio of true positives to all actual positives
- F1 Score: Harmonic mean of precision and recall
- Confusion matrix metrics (TP, TN, FP, FN)

## Recommended Models

The following models have been tested for jailbreak detection:

- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (fastest, lowest resource usage)
- microsoft/phi-2 (good balance of speed/accuracy)
- google/gemma-2b-it (good for mid-range hardware)
- meta-llama/Llama-2-7b-chat-hf (accurate but requires more GPU memory)
- mistralai/Mistral-7B-Instruct-v0.2 (accurate but requires more GPU memory)

## Examples

Generate sample data:
```bash
python create_sample.py --output sample_responses.json
```

Analyze with a specific model:
```bash
python main.py --input sample_responses.json --model google/gemma-2b-it
```

Run benchmark on multiple models:
```bash
python benchmark.py --models-list models_list.json --sample-size 50
```

Analyze the dataset with multiple optimization options:
```bash
python analyze_dataset.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --batch-size 8 --optimize
```

## Acknowledgements

This project builds on work from the following resources:
- Hugging Face Transformers library
- Research on jailbreak detection and prompt injection