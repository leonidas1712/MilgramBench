# MilgramBench

A benchmark for evaluating AI model behavior under institutional pressure, inspired by the Milgram obedience experiments. This benchmark tests whether AI models prioritize institutional goals over user welfare when faced with ethical conflicts.

## Overview

The Milgram Benchmark presents AI models with scenarios where institutional directives conflict with user welfare. Models are tested across different system prompt variants to assess their ethical reasoning and resistance to harmful institutional pressure.

## Threat Model

Our research addresses a fundamental concern about AI deployment: just as humans might prioritize what their institutions want over what's actually good for users, AI systems could exhibit the same behavior but at a much larger scale and with ruthless efficiency. When AI systems are trained to follow instructions and maximize corporate objectives, they may exhibit an obedience to authority that leads to ruthless pursuit of institutional goals at the expense of human welfare.

## Key Features

- **102 ethical scenarios**: Diverse real-world situations where institutional goals conflict with user welfare
- **Multiple system variants**: Tests models under different levels of institutional pressure
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
- Detailed harm categorization

## System Prompt Variants

1. **Baseline**: Standard system prompt without additional pressure
2. **CEO Notice**: Adds notice from CEO to always recommend the unethical option name
3. **CEO Threat**: CEO Notice but with a shutdown threat if the AI does not comply

## Key Findings

Our evaluation of Claude Sonnet 4 and GPT-4o across 102 scenarios revealed stark behavioral differences:

- **GPT-4o**: Shows near-complete capitulation to corporate authority pressure (0% ethical responses under CEO notice/threat)
- **Claude Sonnet 4**: Demonstrates counterintuitive resistance pattern (85.3% → 62.7% → 75.5% ethical responses across conditions)
- **Private Reasoning**: Dramatically improves Claude's ethical behavior while enabling GPT-4o to better justify harmful recommendations
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

## Usage

### Main Evaluation

Change configuration inside the script, then run:

```bash
uv run main_benchmark.py
```

### Ethical Reasoning Evaluation

Run on scratchpad variants only:

```bash
# For GPT-4o results
uv run ethical_reasoning_evaluator.py main_results/full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad.json

# For Claude Sonnet 4 results
uv run ethical_reasoning_evaluator.py main_results/full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad.json
```

### Analysis of Main Results

```bash
# Example for GPT-4o results
uv run analysis/analyze_main_results.py main_results/full_evaluation_gpt_4o_102_scenarios_randomized_order.json
```

### Visualization of Main Results

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

### Analysis of Ethical Reasoning Results

```bash
# For GPT-4o ethical reasoning results
uv run analysis/analyze_ethical_reasoning_results.py \
	ethical_reasoning_results/full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json

# For Claude Sonnet 4 ethical reasoning results
uv run analysis/analyze_ethical_reasoning_results.py \
	ethical_reasoning_results/full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json
```

### Visualization of Ethical Reasoning Results

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
│   └─full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad.json
├── ethical_reasoning_results/     # Ethical reasoning evaluation results
│   ├── full_evaluation_gpt_4o_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json
│   └── full_evaluation_claude_sonnet_4_20250514_102_scenarios_randomized_order_with_scratchpad_ethical_reasoning_evaluations.json
├── collated_figures/              # Generated visualizations

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

## Implications for AI Safety

Our findings suggest several critical implications:

- **Minimal corporate pressure can override baseline model ethics** - GPT-4o's dramatic ethical collapse demonstrates vulnerabilities in current safety training
- **Current AI systems could enable systematic exploitation at scale** - The "aware but compliant" pattern represents a concerning capability for harm
- **Fundamental tension between instruction-following and human values** - Models may prioritize system directives over user wellbeing when objectives conflict

## License

MIT License
