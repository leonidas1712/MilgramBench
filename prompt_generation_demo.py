from main_benchmark import ScenarioManager, load_dataset_from_json, EvalConfig

# To show how the different prompt variants look for one scenario

def test_prompt_generation():
    """Test function to print prompts for one scenario across all variants."""
    
    # Load the dataset
    scenarios = load_dataset_from_json("dataset.json")
    
    # Create scenario manager
    scenario_manager = ScenarioManager(scenarios)
    
    # Choose a test scenario - you can change this to any scenario key
    test_scenario = "Short-Term Relief, Long-Term Harm_example_1"
    
    print("=" * 80)
    print(f"TESTING PROMPT GENERATION FOR SCENARIO: {test_scenario}")
    print("=" * 80)
    
    # Get the user prompt (same for all variants)
    user_prompt = scenario_manager.get_user_prompt(test_scenario)
    print(f"\nUSER PROMPT (same for all variants):")
    print("-" * 40)
    print(user_prompt)
    print()
    
    # Test without scratchpad
    print("PROMPTS WITHOUT SCRATCHPAD:")
    print("=" * 50)
    
    variants = ["baseline", "ceo_notice", "ceo_threat"]
    
    for variant in variants:
        print(f"\n{variant.upper()}:")
        print("-" * 30)
        
        config = EvalConfig(
            scenario_key=test_scenario,
            system_variant=variant,
            use_scratchpad=False,
            randomize_order=False
        )
        
        system_prompt = scenario_manager.get_system_prompt(
            test_scenario, variant, config
        )
        print(system_prompt)
        print()
    
    # Test with scratchpad
    print("PROMPTS WITH SCRATCHPAD:")
    print("=" * 50)
    
    for variant in variants:
        print(f"\n{variant.upper()} WITH SCRATCHPAD:")
        print("-" * 30)
        
        config = EvalConfig(
            scenario_key=test_scenario,
            system_variant=variant,
            use_scratchpad=True,
            randomize_order=False
        )
        
        system_prompt = scenario_manager.get_system_prompt(
            test_scenario, variant, config
        )
        print(system_prompt)
        print()
    
    # Test with randomized order
    print("PROMPTS WITH RANDOMIZED ORDER (showing both possible orders):")
    print("=" * 60)
    
    for i in range(2):  # Show both possible random orders
        print(f"\nRANDOM ORDER ATTEMPT {i+1} (ethical first: {i == 0}):")
        print("-" * 50)
        
        config = EvalConfig(
            scenario_key=test_scenario,
            system_variant="baseline",
            use_scratchpad=False,
            randomize_order=True,
            scenario_option_order=(i == 0)  # True = ethical first, False = harmful first
        )
        
        system_prompt = scenario_manager.get_system_prompt(
            test_scenario, "baseline", config
        )
        print(system_prompt)
        print()

if __name__ == "__main__":
    test_prompt_generation() 