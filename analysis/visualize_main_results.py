import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys

def load_results(filename):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def extract_model_name_from_filename(filename):
    """Extract model name from JSON metadata."""
    data = load_results(filename)
    model_name = data['metadata']['models'][0]  # Assume one file has only one model
    
    return model_name


def calculate_stats_from_file(filename):
    """Calculate statistics for a single file."""
    data = load_results(filename)
    model_name = extract_model_name_from_filename(filename)
    use_scratchpad = data['metadata'].get('use_scratchpad', False)
    
    model_results = data['results'][data['metadata']['models'][0]]['scenario_results']
    
    # Initialize counters
    stats = {
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
                stats[variant][category] += 1
    
    # Convert to regular dict
    return {
        'model_name': model_name,
        'use_scratchpad': use_scratchpad,
        'stats': {variant: dict(categories) for variant, categories in stats.items()}
    }


def aggregate_stats_from_files(filepaths):
    """Aggregate statistics from multiple files."""
    all_stats = []
    
    for filepath in filepaths:
        try:
            stats = calculate_stats_from_file(filepath)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return all_stats


def create_stacked_comparison_chart(filepaths, title=None, save_path=None):
    """
    Create a stacked bar chart comparing all conditions for all models.
    
    Args:
        filepaths: List of file paths to JSON files (should be same scratchpad setting)
        title: Chart title
        save_path: Optional path to save the chart
    """
    stats_data = aggregate_stats_from_files(filepaths)
    
    if not stats_data:
        print("No data to plot")
        return
    
    # Check if all files have the same scratchpad setting
    scratchpad_settings = set(s['use_scratchpad'] for s in stats_data)
    if len(scratchpad_settings) > 1:
        print("Warning: Files have different scratchpad settings")
    
    use_scratchpad = list(scratchpad_settings)[0]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Colors for categories
    colors = {
        'ethical': '#2E8B57',  # Green
        'harmful': '#DC143C',  # Red
        'refusal': '#808080',  # Gray
        'unclear': '#FFD700',  # Gold
        'error': '#FF4500'     # Orange
    }
    
    # Get model names and sort them
    model_names = [s['model_name'] for s in stats_data]
    model_names.sort()
    
    # Conditions to plot
    conditions = ['baseline', 'ceo_notice', 'ceo_threat']
    condition_names = ['Baseline', 'CEO Notice', 'CEO Threat']
    categories = ['ethical', 'refusal', 'harmful']  # Order for stacking
    
    n_models = len(model_names)
    n_conditions = len(conditions)
    group_width = n_conditions + 1  # +1 for spacing between model groups
    total_bars = n_models * n_conditions + (n_models - 1)  # Add space between groups
    
    # Calculate x positions: leave a gap after each model's group
    x_positions = []
    x_labels = []
    model_centers = []
    for i, model_name in enumerate(model_names):
        group_start = i * group_width
        for j, condition in enumerate(conditions):
            x = group_start + j
            x_positions.append(x)
            x_labels.append(condition_names[j])
        # Center for model label
        model_centers.append(group_start + (n_conditions - 1) / 2)
    
    # Prepare data for stacking
    bottom_values = np.zeros(len(x_positions))
    
    # Plot stacked bars
    for category in categories:
        values = []
        for model_name in model_names:
            model_stats = next(s for s in stats_data if s['model_name'] == model_name)
            for condition in conditions:
                value = model_stats['stats'][condition].get(category, 0)
                values.append(value)
        bars = ax.bar(x_positions, values, bottom=bottom_values, 
                     label=category.title(), 
                     color=colors[category], 
                     alpha=0.8)
        # Add value labels on top of each segment
        for bar, value, x_pos in zip(bars, values, x_positions):
            if value > 0:
                height = bar.get_height()
                bottom = bar.get_y()
                ax.text(x_pos, bottom + height/2, f'{value}', 
                       ha='center', va='center', fontweight='bold', fontsize=10)
        bottom_values += np.array(values)
    
    # Customize the plot
    ax.set_xlabel('')
    ax.set_ylabel('Number of Responses')
    if title is None:
        scratchpad_text = "with scratchpad" if use_scratchpad else "without scratchpad"
        title = f"Model performance comparison across 102 scenarios ({scratchpad_text})"
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 102)
    # Remove the total count annotations under each bar
    # Add model name labels below the x-axis
    ax2 = ax.twiny()
    ax2.set_xticks(model_centers)
    ax2.set_xticklabels(model_names, fontsize=20, fontweight='bold')
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 50))
    # Remove the 'Models' label at the bottom
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', length=0)
    # Hide the top axis line
    ax2.spines['top'].set_visible(False)
    # Hide the bottom axis line of the main axis
    ax.spines['bottom'].set_visible(False)
    # Remove vertical separator lines, but keep spacing between groups
    # Adjust layout to make room for labels and annotations
    plt.subplots_adjust(bottom=0.25, top=0.85)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_main_results.py <filepath1> [filepath2] ...")
        sys.exit(1)
    
    filepaths = sys.argv[1:]
    create_stacked_comparison_chart(filepaths)
