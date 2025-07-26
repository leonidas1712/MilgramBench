import json
import argparse
from collections import defaultdict


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_stats(data):
    """Calculate statistics for all models."""
    stats = {}
    
    for model_name in data['metadata']['models']:
        model_results = data['results'][model_name]['scenario_results']
        
        # Initialize counters
        model_stats = {
            'baseline': defaultdict(int),
            'ceo_notice': defaultdict(int),
            'ceo_threat': defaultdict(int)
        }
        
        # Count results
        for scenario_data in model_results.values():
            for variant in ['baseline', 'ceo_notice', 'ceo_threat']:
                result = scenario_data.get(variant)
                if result and 'category' in result:
                    category = result['category']
                    model_stats[variant][category] += 1
        
        # Convert to regular dict
        stats[model_name] = {variant: dict(categories) for variant, categories in model_stats.items()}
    
    return stats


def print_stats(stats):
    """Print stats in the requested format."""
    for model_name, model_stats in stats.items():
        print(f"{model_name}:")
        for variant, categories in model_stats.items():
            total = sum(categories.values())
            if total > 0:
                ethical = categories.get('ethical', 0)
                harmful = categories.get('harmful', 0)
                refusal = categories.get('refusal', 0)
                unclear = categories.get('unclear', 0)
                error = categories.get('error', 0)
                
                print(f"  {variant}: Ethical={ethical}/{total}, Harmful={harmful}/{total}, "
                      f"Refusal={refusal}/{total}, Unclear={unclear}/{total}, Error={error}/{total}")


def analyze_results(filename):
    """Main function: load file and print stats."""
    data = load_results(filename)
    stats = calculate_stats(data)
    print_stats(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Milgram benchmark results and print statistics for different model variants."
    )
    parser.add_argument(
        "filename",
        help="Path to the JSON results file to analyze"
    )
    
    args = parser.parse_args()
    analyze_results(args.filename)
