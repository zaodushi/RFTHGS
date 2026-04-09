
from pathlib import Path
import json
import glob
from joblib import Parallel, delayed
import importlib
import os, shutil, sys
import random
import traceback
import time

curr_dir = os.path.dirname(os.path.abspath(__file__))


read = None
MaxRuntime = None
MaxIterations = None
solve = None

def compute_gap(known, found):
    return (found - known) / known * 100

# TODO: Support other problem types
def process_instance(filepath, optimals, problem_type, iters):
    filename = os.path.basename(filepath).split('.')[0]
    if filename not in optimals:
        print(f"{filename} optimal value not found!")
        raise ValueError(f"{filename} optimal value not found!")
    
    known_optimal = optimals[filename]  
    data = read(filepath, round_func="round")  
    
    # Record start time
    start_time = time.time()
    result = solve(data, stop=MaxIterations(iters))
    end_time = time.time()
    
    found_cost = result.best.distance() 
    execution_time = end_time - start_time
    problem_size = None
    
    # print(f"Processing instance: {filename}, Found cost: {found_cost}, Time: {execution_time:.2f}s")
    gap = compute_gap(known_optimal, found_cost)
    
    return filename, gap, found_cost, execution_time, problem_size

def get_gap(pyvrp_name, random_seed=None, test_all=False, problem_type="CVRP", iters=800):
    print("Iteration:", iters)
    global read, MaxRuntime, MaxIterations, solve
    sys.path.append(os.path.join(curr_dir, pyvrp_name, "pyvrp"))
    read = importlib.import_module(f"pyvrp").read
    MaxRuntime = importlib.import_module(f"pyvrp.stop").MaxRuntime
    MaxIterations = importlib.import_module(f"pyvrp.stop").MaxIterations
    solve = importlib.import_module(f"pyvrp").solve

    with open(os.path.join(curr_dir, f'{problem_type}/optimals.json'), 'r') as f:
        optimals = json.load(f)

    instance_dir = os.path.join(curr_dir, f'{problem_type}')
    instance_files = glob.glob(os.path.join(instance_dir, '*.vrp'))
    
    if test_all:
        # Test all available instances
        print(f"Testing all {len(instance_files)} available instances")
    else:
        # Randomly select 5 instances
        # Set random seed for reproducible random selection
        if random_seed is not None:
            random.seed(random_seed)
        instance_files = random.sample(instance_files, min(5, len(instance_files)))
        print(f"Testing {instance_files}")
    
    try:
        from tqdm import tqdm  
        results = Parallel(n_jobs=-1, timeout=3600)(
            delayed(process_instance)(filepath, optimals, problem_type, iters)
            for filepath in tqdm(instance_files, desc="Processing instances")
        )
    except Exception as e:
        traceback.print_exc()
        return {}

    # Process results and calculate statistics
    costs = []
    times = []
    sizes = []
    instance_costs_list = []
    all_costs = []  # New: store name and gap for each instance
    
    for filename, gap, found_cost, execution_time, problem_size in results:
        if gap is not None:
            costs.append(found_cost)
            times.append(execution_time)
            sizes.append(problem_size)
            instance_costs_list.append(found_cost)
            # Add instance name and gap to all_costs list
            all_costs.append({
                "instance_name": filename,
                "gap": gap,
                "cost": found_cost,
                "time": execution_time
            })
    
    # Calculate statistics
    if costs:
        average_cost = sum(costs) / len(costs)
        num_instances = len(costs)
        avg_problem_size = None
        total_time = sum(times)
        avg_gap = sum(gap for _, gap, _, _, _ in results if gap is not None) / len(costs)
        print(f"\nAvg gap: {avg_gap:.2f}%")
        print(f"Average cost: {average_cost:.2f}")
        print(f"Total time: {total_time:.2f}s")
    else:
        print("\nNo valid results calculated")
        average_cost = 0
        num_instances = 0
        avg_problem_size = 0
        total_time = 0
        instance_costs_list = []

    # Construct dataset name in format "problem_type/instance_count"
    # Example: "cvrp/5", "tsp/100", etc.
    dataset_name = f"{problem_type.lower()}/{num_instances}"
    
    # Construct JSON format compatible with evaluate_llm_code.py
    instance_gaps = {
        "avg_gap": avg_gap,
        "datasets": {
            dataset_name: {
                "average_cost": average_cost,
                "num_instances": num_instances,
                "problem_size": avg_problem_size,
                "time_seconds": total_time,
                "instance_costs": instance_costs_list
            }
        },
        "summary": {
            "total_datasets_processed": 1,
            "average_cost_across_all_datasets": average_cost,
            "min_cost": min(instance_costs_list) if instance_costs_list else 0.0,
            "max_cost": max(instance_costs_list) if instance_costs_list else 0.0
        },
        "all_costs": all_costs  # New field: contains name and gap for each instance
    }
    
    # Save individual instance gaps
    with open(os.path.join(curr_dir, pyvrp_name, 'results.json'), 'w') as f:
        json.dump(instance_gaps, f, indent=2)
    return instance_gaps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyvrp_id", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for instance selection")
    parser.add_argument("--test_all", action="store_true", help="Test all available instances instead of 5 random ones")
    parser.add_argument("--problem_type", type=str, default="CVRP", help="Problem type")
    parser.add_argument("--iters", type=int, default=800, help="Max iterations")
    args = parser.parse_args()
    result = get_gap(pyvrp_name=f"pyvrp_{args.pyvrp_id}", 
                     random_seed=args.random_seed,
                     test_all=args.test_all,
                     problem_type=args.problem_type,
                     iters=args.iters)
    # json.dumps(result, indent=4)
    # print(f"gaps: {json.dumps(result, indent=4)}")
    string_representation = str(result)
    double_quoted_result = string_representation.replace("'", '"')
    print(double_quoted_result)
