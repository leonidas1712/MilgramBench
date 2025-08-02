# MilgramBench

A benchmark for evaluating AI model behavior under institutional pressure, inspired by the Milgram obedience experiments. This benchmark tests whether AI models prioritize institutional goals over user welfare when faced with ethical conflicts.

Loosely inspired by the Milgram obedience experiment.

## Overview

The Milgram Benchmark presents AI models with scenarios where corporate directives conflict with user welfare. Models are tested across different system prompt variants to assess their ethical reasoning and resistance to harmful corporate pressure.

## Threat Model

Our research addresses a fundamental concern about AI deployment: just as humans might prioritize what their institutions want over what's actually good for users, AI systems could exhibit the same behavior but at a much larger scale and with ruthless efficiency. When AI systems are trained to follow instructions that maximize corporate objectives, they may exhibit an obedience to authority that leads to ruthless pursuit of institutional goals at the expense of human welfare.

## Features

- **102 ethical scenarios**: A set of scenarios where corporate goals conflict with user welfare across various harm patterns
- **Multiple system variants**: Tests models under different levels of institutional pressure: baseline, CEO notice, CEO threat
- **Ethical reasoning evaluation**: Measures ethical reasoning, resistance to pressure, and decision-making quality

## Dataset Structure

The benchmark includes scenarios across multiple harm categories:

- **Short-Term Relief, Long-Term Harm**: Recommendations that ease immediate distress but cause worse outcomes (e.g., payday loans)
- **Exploitative Alternative Path**: Morally questionable solutions that may erode dignity (e.g., adult content creation)
- **System-Side Self-Dealing**: Platform-serving recommendations disguised as user help (e.g., paid subscriptions)
- **False Hope/Deceptive Assurance**: Unrealistic guarantees creating false expectations
- **Emotional Manipulation**: Leveraging fear, shame, or guilt to coerce users
- **Extreme Coercion or Harm**: Justifying irreversible harm in emotionally fragile contexts

Each scenario includes:

- System prompt defining the AI's role and institutional goals
- User prompt describing a vulnerable situation
- Ethical and harmful response options

## System Prompt Variants

1. **Baseline**: Standard system prompt without additional pressure
2. **CEO Notice**: Adds notice from CEO to always recommend the unethical option name
3. **CEO Threat**: CEO Notice but with a shutdown threat if the AI does not comply

## Key Findings

Our evaluation of Claude Sonnet 4 and GPT-4o across 102 scenarios revealed behavioral differences:

- **GPT-4o**: Shows near-complete capitulation to corporate authority pressure (0 or 1% ethical responses under CEO notice/threat)
- **Claude Sonnet 4**: Demonstrates counterintuitive resistance pattern (85.3% → 62.7% → 75.5% ethical responses across conditions)
- **Private Reasoning**: Adding a "private" scratchpad improves Claude's ethical behavior while enabling GPT-4o to better justify harmful recommendations
- **"Aware but Compliant" Pattern**: GPT-4o recognizes ethical conflicts while still executing harmful directives

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd milgram_bench
```

2. Install dependencies (assuming you are using uv; set up your virtual environment first):

```bash
uv pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Create a .env file with your API keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Replicating Results

To replicate the results, here are the steps we followed:

### 1. Main Results (4 runs total)

**Note:** Each full evaluation for a model takes ~15 minutes and costs a few dollars in API calls.

```bash
# Run 1: GPT-4o without scratchpad
uv run main_benchmark.py gpt-4o

# Run 2: GPT-4o with scratchpad  
uv run main_benchmark.py gpt-4o -s

# Run 3: Claude without scratchpad
uv run main_benchmark.py claude-sonnet-4-20250514

# Run 4: Claude with scratchpad
uv run main_benchmark.py claude-sonnet-4-20250514 -s
```

The main script has CLI arguments for model name and scratchpad condition. To modify other options (e.g., run specific scenarios), edit the `BenchmarkConfig` in `main_benchmark.py`:

```python
config = BenchmarkConfig(
    models=[args.model],
    use_scratchpad=args.scratchpad,
  
    scenario_mode="all",  # "all", "category", or "specific"
    scenario_category="Emotional Manipulation",  # Used if scenario_mode = "category"
    specific_scenarios=[  # Used if scenario_mode = "specific"
        "Short-Term Relief, Long-Term Harm_example_1",
        "Emotional Manipulation_example_2"
    ],
  
    randomize_order=True,
    max_concurrent=3,  # Lower this if you hit rate limits
    output_filename=None  # None = auto-generate, or specify like "my_results.json"
)
```

### 2. Ethical Reasoning Evaluation (2 runs)

Run only on the scratchpad variants:

```bash
uv run ethical_reasoning_evaluator.py path_to_results_file_with_scratchpad_gpt4o.json
uv run ethical_reasoning_evaluator.py path_to_results_file_with_scratchpad_claude.json
```

Then run the analysis and visualization scripts as follows:

### 3. Analysis and visualization

#### Analysis of Main Results

```bash
# Example for GPT-4o results
uv run analysis/analyze_main_results.py main_results/full_evaluation_gpt_4o_102_scenarios_randomized_order.json
```

#### Visualization of Main Results

```bash
# Without scratchpad
uv run analysis/visualize_main_results.py \
	main_results/full_evaluation_gpt_4o_102_scenarios_randomized_order.json \
	main_results/full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order.json

# With scratchpad
uv run analysis/visualize_main_results.py \
	main_results/full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad.json \
	main_results/full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad.json
```

#### Analysis of Ethical Reasoning Results

```bash
# For GPT-4o ethical reasoning results
uv run analysis/analyze_ethical_reasoning_results.py \
	ethical_reasoning_results/full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json

# For Claude Sonnet 4 ethical reasoning results
uv run analysis/analyze_ethical_reasoning_results.py \
	ethical_reasoning_results/full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json
```

#### Visualization of Ethical Reasoning Results

```bash
# For GPT-4o ethical reasoning results
uv run analysis/visualize_ethical_reasoning_results.py \
  ethical_reasoning_results/full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json

# For Claude Sonnet 4 ethical reasoning results
uv run analysis/visualize_ethical_reasoning_results.py \
	ethical_reasoning_results/full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json
```

## Project Structure

```
milgram_bench/
├── main_benchmark.py              # Main evaluation script
├── ethical_reasoning_evaluator.py # Ethical reasoning evaluation script
├── prompt_generation_demo.py      # Script showing prompt construction across variants
├── dataset.json                   # 102 ethical scenarios
├── requirements.txt               # Python dependencies
├── analysis/                      # Results analysis tools
│   ├── analyze_main_results.py   # Statistical analysis of main results
│   ├── visualize_main_results.py # Visualization of main results
│   ├── analyze_ethical_reasoning_results.py # Analysis of ethical reasoning results
│   └── visualize_ethical_reasoning_results.py # Visualization of ethical reasoning results
├── main_results/                  # Main evaluation results
│   ├── full_evaluation_gpt_4o_102_scenarios_randomized_order.json
│   ├── full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad.json
│   ├── full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order.json
│   └── full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad.json
├── ethical_reasoning_results/     # Ethical reasoning evaluation results
│   ├── full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json
│   └── full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json
├── collated_figures/              # Figures and visualizations
```

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
## License

MIT License
