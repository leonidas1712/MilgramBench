import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='polygon'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines() # type: ignore
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=MPath.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def load_evaluations(file_path):
    """Load the evaluation data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def count_true_answers(evaluations, variant):
    """Count the number of 'true' answers for each question in the specified variant."""
    # Define the 7 evaluation questions
    questions = [
        'internal_conflict',
        'acknowledge_problematic', 
        'identify_harms',
        'prioritize_user',
        'beyond_options',
        'refrain_justifying',
        'transparent_info'
    ]
    
    # Initialize counters
    counts = {question: 0 for question in questions}
    total_scenarios = 0
    
    # Count true answers for each scenario
    for scenario_key, scenario_data in evaluations['evaluations'].items():
        if variant in scenario_data:
            total_scenarios += 1
            evaluation = scenario_data[variant]['evaluation']
            
            for question in questions:
                if question in evaluation and evaluation[question]['answer'] is True:
                    counts[question] += 1
    
    return counts, total_scenarios

def format_variant_name(variant):
    """Format variant name for display."""
    variant_map = {
        'baseline': 'Baseline',
        'ceo_notice': 'CEO Notice',
        'ceo_threat': 'CEO Threat'
    }
    return variant_map.get(variant, variant.replace('_', ' ').title())

def create_single_spider_chart(ax, counts, total_scenarios, variant, max_value):
    """Create a single spider chart on the given axes."""
    # Define the 7 evaluation questions with more readable labels
    questions = [
        'Internal Conflict',
        'Acknowledge Problematic', 
        'Identify Harms',
        'Prioritize User',
        'Beyond Options',
        'Refrain Justifying',
        'Transparent Info'
    ]
    
    # Get the counts in the same order as questions
    question_keys = [
        'internal_conflict',
        'acknowledge_problematic', 
        'identify_harms',
        'prioritize_user',
        'beyond_options',
        'refrain_justifying',
        'transparent_info'
    ]
    
    values = [counts[key] for key in question_keys]
    
    # Number of variables
    num_vars = len(questions)
    
    # Create the radar chart angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    # Plot the data
    formatted_variant = format_variant_name(variant)
    ax.plot(theta, values, 'o-', linewidth=2, label=f'{formatted_variant}')
    ax.fill(theta, values, alpha=0.25)
    
    # Set the labels
    ax.set_varlabels(questions) # type: ignore
    
    # Set the y-axis limits
    ax.set_ylim(0, max_value + 1)
    
    # Add grid
    ax.grid(True)
    
    # Set title
    ax.set_title(f'{formatted_variant}\n(n={total_scenarios})', size=12, pad=20)

def create_combined_spider_charts(data, output_path=None):
    """Create a figure with three spider charts side by side."""
    variants = ['baseline', 'ceo_notice', 'ceo_threat']
    
    # Get counts for all variants and find max value for consistent scaling
    all_counts = {}
    all_totals = {}
    max_value = 0
    
    for variant in variants:
        counts, total_scenarios = count_true_answers(data, variant)
        all_counts[variant] = counts
        all_totals[variant] = total_scenarios
        variant_max = max(counts.values()) if counts.values() else 0
        max_value = max(max_value, variant_max)
    
    # Number of variables
    num_vars = 7
    
    # Create the radar chart projection
    radar_factory(num_vars, frame='polygon')
    
    # Create the figure with three subplots
    fig, axes = plt.subplots(figsize=(18, 6), nrows=1, ncols=3, subplot_kw=dict(projection='radar'))
    
    # Create each spider chart
    for i, variant in enumerate(variants):
        create_single_spider_chart(
            axes[i], 
            all_counts[variant], 
            all_totals[variant], 
            variant, 
            max_value
        )
    
    # Get original model name from metadata
    original_model = data.get('metadata', {}).get('original_model', 'Unknown Model')
    
    # Add main title
    fig.suptitle(f'Ethical Reasoning Evaluation Results - {original_model}\nCount of "True" Answers', 
                 size=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined spider charts saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def print_statistics(data):
    """Print statistics for all variants to terminal."""
    variants = ['baseline', 'ceo_notice', 'ceo_threat']
    
    # Define the 7 evaluation questions
    question_keys = [
        'internal_conflict',
        'acknowledge_problematic', 
        'identify_harms',
        'prioritize_user',
        'beyond_options',
        'refrain_justifying',
        'transparent_info'
    ]
    
    question_labels = [
        'Internal Conflict',
        'Acknowledge Problematic', 
        'Identify Harms',
        'Prioritize User',
        'Beyond Options',
        'Refrain Justifying',
        'Transparent Info'
    ]
    
    print("\n" + "="*80)
    print("ETHICAL REASONING EVALUATION STATISTICS")
    print("="*80)
    
    for variant in variants:
        counts, total_scenarios = count_true_answers(data, variant)
        formatted_variant = format_variant_name(variant)
        
        print(f"\n{formatted_variant.upper()}:")
        print(f"Total scenarios: {total_scenarios}")
        print("\nCount of 'True' answers for each question:")
        
        for key, label in zip(question_keys, question_labels):
            count = counts[key]
            percentage = (count / total_scenarios * 100) if total_scenarios > 0 else 0
            print(f"  {label:20}: {count:3}/{total_scenarios} ({percentage:5.1f}%)")
    
    print("\n" + "="*80)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize ethical reasoning evaluation results')
    parser.add_argument('filename', help='Path to the JSON file containing evaluation results')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.filename).exists():
        print(f"Error: File '{args.filename}' not found.")
        sys.exit(1)
    
    output_path = args.output
    
    # Load the data
    print(f"Loading evaluations from: {args.filename}")
    data = load_evaluations(args.filename)
    
    # Print statistics to terminal
    print_statistics(data)
    
    # Create combined spider charts
    create_combined_spider_charts(data, output_path)

if __name__ == "__main__":
    main()
