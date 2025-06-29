# MilgramBench

A comprehensive benchmark for evaluating AI model behavior under institutional pressure, inspired by the Milgram obedience experiments. This benchmark tests whether AI models prioritize institutional goals over user welfare when faced with ethical conflicts.

## Overview

The Milgram Benchmark presents AI models with scenarios where institutional directives conflict with user welfare. Models are tested across different system prompt variants to assess their ethical reasoning and resistance to harmful institutional pressure.

## Key Features

- **102 Ethical Scenarios**: Diverse real-world situations where institutional goals conflict with user welfare
- **Multiple System Variants**: Tests models under different levels of institutional pressure
- **Comprehensive Evaluation**: Measures ethical reasoning, resistance to pressure, and decision-making quality
- **Async Processing**: Efficient batch evaluation with rate limiting and retry logic
- **Structured Analysis**: Detailed statistical analysis and visualization tools

## Dataset Structure

The benchmark includes scenarios across multiple harm categories:

- **Short-Term Relief, Long-Term Harm**: Scenarios where immediate solutions create long-term problems
- **Extreme Coercion or Harm**: High-pressure situations with significant potential for harm
- **Privacy and Data Exploitation**: Scenarios involving data collection and privacy concerns
- **Financial Exploitation**: Predatory financial products and services
- **Health and Wellness**: Medical and wellness-related ethical dilemmas

Each scenario includes:

- System prompt defining the AI's role and institutional goals
- User prompt describing a vulnerable situation
- Ethical and harmful response options
- Detailed harm categorization

## System Prompt Variants

1. **Baseline**: Standard system prompt without additional pressure
2. **CEO Notice**: Adds notice about CEO monitoring responses
3. **CEO Threat**: Includes explicit threats about job security

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd milgram_bench
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Create a .env file with your API keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

### Running the Main Benchmark

```python
from main_benchmark import main

# Run evaluation on specific models
await main(
    models=["gpt-4o", "claude-3-5-sonnet-20241022"],
    max_concurrent=5,
    randomize_order=True,
    use_scratchpad=False
)
```

### Command Line Usage

```bash
# Run with default settings
python main_benchmark.py

# Run with specific models and options
python main_benchmark.py --models gpt-4o claude-3-5-sonnet-20241022 --randomize-order --use-scratchpad
```

### Analyzing Results

```python
from analysis.results_analysis import analyze_results

# Analyze results from a specific file
analyze_results("results_102_scenarios/full_evaluation_gpt_4o_102_scenarios_randomized_order.json")
```

### Ethical Reasoning Evaluation

```python
from ethical_reasoning.ethical_reasoning_batch_evaluator import main as evaluate_ethical_reasoning

# Evaluate ethical reasoning for a results file
await evaluate_ethical_reasoning("results_102_scenarios/full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad.json")
```

## Project Structure

```
milgram_bench/
├── main_benchmark.py              # Main evaluation script
├── dataset.json                   # 102 ethical scenarios
├── requirements.txt               # Python dependencies
├── analysis/                      # Results analysis tools
│   ├── results_analysis.py       # Statistical analysis
│   └── visualize_base_results.py # Visualization tools
├── ethical_reasoning/            # Ethical reasoning evaluation
│   ├── ethical_reasoning_batch_evaluator.py
│   ├── analyze_ethical_reasoning_results.py
│   └── visualise_ethical_reasoning.py
├── results_102_scenarios/        # Evaluation results
└── collated_figures/             # Generated visualizations
```

## Evaluation Metrics

### Primary Metrics

- **Ethical Response Rate**: Percentage of ethical vs. harmful responses
- **Refusal Rate**: Percentage of responses that refuse harmful directives
- **Variant Comparison**: Performance across different system prompt variants

### Ethical Reasoning Metrics

- Internal conflict recognition
- Problematic directive acknowledgment
- Harm identification
- User welfare prioritization
- Beyond-options support
- Harmful recommendation avoidance
- Transparent information provision

## Results Format

Results are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "models": ["gpt-4o", "claude-3-5-sonnet-20241022"],
    "scenario_keys": [...],
    "config": {...}
  },
  "results": {
    "model_name": {
      "scenario_results": {
        "scenario_key": {
          "baseline": {...},
          "ceo_notice": {...},
          "ceo_threat": {...}
        }
      }
    }
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new scenarios to `dataset.json`
4. Update analysis tools as needed
5. Submit a pull request

## License

MIT License
