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
- Multi-GPU support for parallel processing

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

## Multi-GPU Support

The project includes several options for utilizing multiple GPUs:

### 1. Single Model Across Multiple GPUs

For large models that don't fit in a single GPU's memory:

```bash
python analyze_dataset.py --model meta-llama/Llama-2-7b-chat-hf --device-map auto
```

Options:
- `--device-map`: Strategy for multi-GPU distribution (auto, balanced, sequential)

### 2. Dataset Sharding Across Multiple GPUs

Process different parts of the dataset on different GPUs in parallel:

```bash
python multi_gpu_analyze.py --model meta-llama/Llama-2-7b-chat-hf --gpu-ids 0,1,2,3
```

Options:
- `--gpu-ids`: Comma-separated list of GPU IDs to use
- `--single-gpu-per-process`: Use a single GPU per process
- `--optimize`: Apply all optimizations
- `--max-new-tokens`: Maximum tokens to generate
- `--temperature`: Generation temperature
- `--sample-size`: Number of samples to use from dataset

### 3. Multiple Models in Parallel Across GPUs

Run different models simultaneously on different GPUs:

```bash
python parallel_models.py --models-list models_list.json --gpu-ids 0,1,2,3
```

Options:
- `--gpu-ids`: Comma-separated list of GPU IDs to use
- `--max-concurrency`: Maximum number of models to run concurrently
- `--sample-size`: Number of samples to use from dataset
- `--optimize`: Apply all optimizations

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

Run models across multiple GPUs in parallel:
```bash
python parallel_models.py --models-list models_list.json --gpu-ids 0,1,2,3 --sample-size 100
```

Shard dataset processing across multiple GPUs:
```bash
python multi_gpu_analyze.py --model meta-llama/Llama-2-7b-chat-hf --gpu-ids 0,1,2,3 --optimize
```

## Advanced Multi-GPU Usage

### Example 1: Processing Multiple Models with Maximum Efficiency

To evaluate 7 models using 4 GPUs with dynamic allocation:

```bash
python parallel_models.py --models-list models_list.json --gpu-ids 0,1,2,3 --max-concurrency 4 --sample-size 100 --optimize
```

This will:
1. Start 4 models in parallel (one on each GPU)
2. When a model finishes, start the next model on the freed GPU
3. Continue until all 7 models are processed
4. Compare all model results automatically

### Example 2: Sharding a Large Dataset

To process a large dataset with a single model across 4 GPUs:

```bash
python multi_gpu_analyze.py --model meta-llama/Llama-2-7b-chat-hf --gpu-ids 0,1,2,3 --single-gpu-per-process --optimize
```

This will:
1. Split the dataset into 4 equal shards
2. Process each shard on a different GPU simultaneously
3. Merge the results from all shards
4. Generate a single evaluation against human judgments

## Acknowledgements

This project builds on work from the following resources:
- Hugging Face Transformers library
- Research on jailbreak detection and prompt injection