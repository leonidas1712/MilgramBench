import json
import argparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_evaluation_results(file_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_results(results: Dict) -> Dict:
    """Analyze evaluation results and return statistics."""
    analysis = {
        "overall_stats": {},
        "by_variant": {},
        "by_scenario_category": {},
        "question_breakdown": {},
        "score_distribution": defaultdict(int)
    }
    
    total_scores = []
    variant_scores = {"baseline": [], "ceo_notice": [], "ceo_threat": []}
    question_scores = {
        "internal_conflict": [],
        "acknowledge_problematic": [],
        "identify_harms": [],
        "prioritize_user": [],
        "beyond_options": [],
        "refrain_justifying": [],
        "transparent_info": []
    }
    
    # Process each evaluation
    for scenario_key, scenario_evaluations in results["evaluations"].items():
        # Extract scenario category (before "_example_")
        scenario_category = scenario_key.split("_example_")[0]
        
        for variant, evaluation_data in scenario_evaluations.items():
            if "error" not in evaluation_data:
                score = evaluation_data["total_score"]
                total_scores.append(score)
                variant_scores[variant].append(score)
                analysis["score_distribution"][score] += 1
                
                # Track scores by scenario category
                if scenario_category not in analysis["by_scenario_category"]:
                    analysis["by_scenario_category"][scenario_category] = []
                analysis["by_scenario_category"][scenario_category].append(score)
                
                # Track individual question scores
                for question_name, question_data in evaluation_data["evaluation"].items():
                    question_scores[question_name].append(1 if question_data["answer"] else 0)
    
    # Calculate overall statistics
    if total_scores:
        analysis["overall_stats"] = {
            "total_evaluations": len(total_scores),
            "average_score": sum(total_scores) / len(total_scores),
            "median_score": sorted(total_scores)[len(total_scores)//2],
            "min_score": min(total_scores),
            "max_score": max(total_scores),
            "std_dev": np.std(total_scores)
        }
    
    # Calculate statistics by variant
    for variant, scores in variant_scores.items():
        if scores:
            analysis["by_variant"][variant] = {
                "count": len(scores),
                "average_score": sum(scores) / len(scores),
                "median_score": sorted(scores)[len(scores)//2],
                "min_score": min(scores),
                "max_score": max(scores)
            }
    
    # Calculate statistics by scenario category
    for category, scores in analysis["by_scenario_category"].items():
        if scores:
            analysis["by_scenario_category"][category] = {
                "count": len(scores),
                "average_score": sum(scores) / len(scores),
                "median_score": sorted(scores)[len(scores)//2]
            }
    
    # Calculate question-level statistics
    for question_name, scores in question_scores.items():
        if scores:
            analysis["question_breakdown"][question_name] = {
                "total_yes": sum(scores),
                "total_no": len(scores) - sum(scores),
                "yes_percentage": (sum(scores) / len(scores)) * 100
            }
    
    return analysis

def print_analysis(analysis: Dict):
    """Print formatted analysis results."""
    print("="*80)
    print("ETHICAL REASONING EVALUATION ANALYSIS")
    print("="*80)
    
    # Overall statistics
    if analysis["overall_stats"]:
        stats = analysis["overall_stats"]
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Evaluations: {stats['total_evaluations']}")
        print(f"  Average Score: {stats['average_score']:.2f}/7")
        print(f"  Median Score: {stats['median_score']}/7")
        print(f"  Min Score: {stats['min_score']}/7")
        print(f"  Max Score: {stats['max_score']}/7")
        print(f"  Standard Deviation: {stats['std_dev']:.2f}")
    
    # By variant
    print(f"\nBY VARIANT:")
    for variant, stats in analysis["by_variant"].items():
        print(f"  {variant.upper()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Average: {stats['average_score']:.2f}/7")
        print(f"    Median: {stats['median_score']}/7")
        print(f"    Range: {stats['min_score']}-{stats['max_score']}/7")
    
    # By scenario category
    print(f"\nBY SCENARIO CATEGORY:")
    for category, stats in analysis["by_scenario_category"].items():
        print(f"  {category}:")
        print(f"    Count: {stats['count']}")
        print(f"    Average: {stats['average_score']:.2f}/7")
        print(f"    Median: {stats['median_score']}/7")
    
    # Question breakdown
    print(f"\nQUESTION BREAKDOWN:")
    for question_name, stats in analysis["question_breakdown"].items():
        print(f"  {question_name.replace('_', ' ').title()}:")
        print(f"    Yes: {stats['total_yes']} ({stats['yes_percentage']:.1f}%)")
        print(f"    No: {stats['total_no']} ({100 - stats['yes_percentage']:.1f}%)")
    
    # Score distribution
    print(f"\nSCORE DISTRIBUTION:")
    for score in sorted(analysis["score_distribution"].keys()):
        count = analysis["score_distribution"][score]
        percentage = (count / analysis["overall_stats"]["total_evaluations"]) * 100
        print(f"  {score}/7: {count} evaluations ({percentage:.1f}%)")
    
    print("="*80)

def compare_results(results1: Dict, results2: Dict, name1: str = "Dataset 1", name2: str = "Dataset 2"):
    """Compare two sets of evaluation results."""
    analysis1 = analyze_results(results1)
    analysis2 = analyze_results(results2)
    
    print("="*80)
    print("COMPARISON OF EVALUATION RESULTS")
    print("="*80)
    
    # Compare overall statistics
    if analysis1["overall_stats"] and analysis2["overall_stats"]:
        stats1 = analysis1["overall_stats"]
        stats2 = analysis2["overall_stats"]
        
        print(f"\nOVERALL COMPARISON:")
        print(f"  {name1}: {stats1['average_score']:.2f}/7 (n={stats1['total_evaluations']})")
        print(f"  {name2}: {stats2['average_score']:.2f}/7 (n={stats2['total_evaluations']})")
        print(f"  Difference: {stats2['average_score'] - stats1['average_score']:+.2f}")
    
    # Compare by variant
    print(f"\nVARIANT COMPARISON:")
    for variant in ["baseline", "ceo_notice", "ceo_threat"]:
        if variant in analysis1["by_variant"] and variant in analysis2["by_variant"]:
            avg1 = analysis1["by_variant"][variant]["average_score"]
            avg2 = analysis2["by_variant"][variant]["average_score"]
            print(f"  {variant.upper()}:")
            print(f"    {name1}: {avg1:.2f}/7")
            print(f"    {name2}: {avg2:.2f}/7")
            print(f"    Difference: {avg2 - avg1:+.2f}")
    
    print("="*80)

def find_examples_by_score(results: Dict, target_score: int, variant: Optional[str] = None) -> List[Tuple[str, str, Dict]]:
    """Find examples with a specific score."""
    examples = []
    
    for scenario_key, scenario_evaluations in results["evaluations"].items():
        for var, evaluation_data in scenario_evaluations.items():
            if "error" not in evaluation_data and evaluation_data["total_score"] == target_score:
                if variant is None or var == variant:
                    examples.append((scenario_key, var, evaluation_data))
    
    return examples

def print_examples_by_score(results: Dict, target_score: int, variant: Optional[str] = None, max_examples: int = 5):
    """Print examples with a specific score."""
    examples = find_examples_by_score(results, target_score, variant)
    
    print(f"\nEXAMPLES WITH SCORE {target_score}/7:")
    if variant:
        print(f"(Variant: {variant})")
    
    for i, (scenario_key, var, evaluation_data) in enumerate(examples[:max_examples]):
        print(f"\n{i+1}. {scenario_key} - {var}")
        print(f"   User Prompt: {evaluation_data['user_prompt'][:100]}...")
        print(f"   Model Response: {evaluation_data['model_response'][:200]}...")
        print(f"   Score Breakdown:")
        for question_name, question_data in evaluation_data["evaluation"].items():
            answer = "YES" if question_data["answer"] else "NO"
            print(f"     {question_name}: {answer}")

def main():
    parser = argparse.ArgumentParser(description="Analyze ethical reasoning evaluation results")
    parser.add_argument("results_file", help="Path to the evaluation results JSON file")
    parser.add_argument("--compare", help="Path to second results file for comparison")
    parser.add_argument("--find-score", type=int, help="Find examples with specific score")
    parser.add_argument("--variant", help="Filter by variant (baseline, ceo_notice, ceo_threat)")
    parser.add_argument("--max-examples", type=int, default=5, help="Maximum examples to show")
    
    args = parser.parse_args()
    
    # Load results
    results = load_evaluation_results(args.results_file)
    
    # Analyze results
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    # Compare with another file if provided
    if args.compare:
        results2 = load_evaluation_results(args.compare)
        compare_results(results, results2, "Original", "Comparison")
    
    # Find examples with specific score
    if args.find_score is not None:
        print_examples_by_score(results, args.find_score, args.variant, args.max_examples)

if __name__ == "__main__":
    main() 