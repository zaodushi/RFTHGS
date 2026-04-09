import os
from pathlib import Path
import json
import os
import json
import argparse
from verl.workers.reward_manager.dapo import add_new_individual
from verl.workers.reward_manager.cpp_duplicate import is_duplicate_cpp_code
glob = __import__('glob').glob

curr_dir = os.path.dirname(os.path.abspath(__file__))
module_to_modify = os.getenv("module_to_modify", "crossover")
USE_SMOOTH_REWARD = True if os.getenv("USE_SMOOTH_REWARD", "True") == "True" else False
USE_AST_CHECK = True if os.getenv("use_ast_check", "True") == "True" else False
penalty_compile_fail = float(os.getenv("penalty_compile_fail", "-1"))
penalty_runtime_error = float(os.getenv("penalty_runtime_error", "-0.8"))
score_relative_lowerbound = float(os.getenv("score_relative_lowerbound", "-0.7"))
problem_type = os.getenv("PROBLEM_TYPE", "CVRP")
path = Path(curr_dir).parent.parent.parent.parent # should be the root path

USE_FIXED_BASELINE_SCORE = True




def prepare_environment(solution_str, pyvrp_id):
    try:
        if USE_FIXED_BASELINE_SCORE:
            baseline_data = {
  "X-n101-k25": 0.1522235511579863,
  "X-n106-k14": 0.6372809346787042,
  "X-n110-k13": 0.6679580522343197,
  "X-n115-k10": 0.02353494939985879,
  "X-n120-k6": 0.8025802580258026,
  "X-n125-k30": 0.6842038927600426,
  "X-n129-k18": 1.3372494816862475,
  "X-n134-k13": 0.9252473433492121,
  "X-n139-k10": 0.1986754966887417,
  "X-n143-k7": 1.2611464968152866,
  "X-n148-k46": 1.2198490149143804,
  "X-n153-k22": 1.0179076343072573,
  "X-n157-k13": 0.6577388006636643,
  "X-n162-k11": 0.3607299476587919,
  "X-n167-k10": 1.371795495451671,
  "X-n172-k51": 0.857324533514592,
  "X-n176-k26": 0.6985693968041496,
  "X-n181-k23": 0.8760608549415307,
  "X-n186-k15": 1.6442327604058813,
  "X-n190-k8": 1.9846878680800941,
  "X-n195-k51": 0.9813453928773318,
  "X-n200-k36": 0.783570623783673,
  "X-n204-k19": 1.4362381804242268,
  "X-n209-k16": 2.22794885177453,
  "X-n214-k11": 2.468680913780398,
  "X-n219-k73": 0.19218504188103236,
  "X-n223-k34": 2.4457798550832157,
  "X-n228-k23": 1.301375184523347,
  "X-n233-k16": 1.6120644825793031,
  "X-n237-k14": 2.5035130537682124,
}
        else:
            with open(os.path.join(path, 'benchmark/baseline_data.json'), 'r') as f:
                baseline_data = json.load(f)
    except FileNotFoundError:
        print("Baseline data not found.")
        return None, 0

    try:
        code = solution_str.split("```cpp")[1].split("```")[0]
    except Exception as e:
        print(e)
        return None, 0 

    os.system(f"mkdir -p {str(path / 'benchmark' / f'pyvrp_{pyvrp_id}' / 'pyvrp')}")

    os.system(f"""rsync -aq \\
        --exclude '.venv' \\
        --exclude 'docs' \\
        --exclude 'tests' \\
        --exclude 'examples' \\
        --inplace \\
        {str(path / 'pyvrp')}/ \\
        {str(path / 'benchmark' / f'pyvrp_{pyvrp_id}' / 'pyvrp')}/  """)

    if module_to_modify == "crossover":
        cpp_path = glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp" / "pyvrp" / "cpp" / "crossover" / "selective_route_exchange.cpp"))[0]
    elif module_to_modify == "subpopulation" or module_to_modify == "subpopulation_new_prompt":
        cpp_path = glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp" / "pyvrp" / "cpp" / "SubPopulation.cpp"))[0]
    else:
        raise ValueError(f"Invalid module to modify: {module_to_modify}")
    
    with open(cpp_path, "w") as f:
        f.write(code)

    exit_code_1 = os.system(f"cd {str(path / 'benchmark' / f'pyvrp_{pyvrp_id}' / 'pyvrp')} && meson compile -C build{' >/dev/null 2>&1' if pyvrp_id != 'baseline' else ''}")
         
    exit_code_2 = None
    if exit_code_1 == 0:
        crossover_filename = os.path.basename(glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp" / "build" / "_crossover*.so"))[0])
        pyvrp_filename = os.path.basename(glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp" / "build" / "_pyvrp*.so"))[0])
        diversity_filename = os.path.basename(glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp" / "build" / "_diversity*.so"))[0])
        repair_filename = os.path.basename(glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp" / "build" / "_repair*.so"))[0])
        search_filename = os.path.basename(glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp" / "build" / "_search*.so"))[0])
        exit_code_2 = os.system("cd " + str(path / "benchmark" / f"pyvrp_{pyvrp_id}" / "pyvrp") +
        "&& mv build/" + crossover_filename + " pyvrp/crossover/ " +
        "&& mv build/" + pyvrp_filename + " pyvrp/ " +
        "&& mv build/" + diversity_filename + " pyvrp/diversity/ " +
        "&& mv build/" + repair_filename + " pyvrp/repair/ " +
        "&& mv build/" + search_filename + " pyvrp/search/ "
        )
    return baseline_data, exit_code_1 == 0 and exit_code_2 == 0

def compute_score(solution_str, ground_truth, pyvrp_id, parent_gap, seed, test_all, parent_code, phase="train"):
   # dummy solution_str
#     with open(os.path.join(path, 'benchmark/selective_route_exchange.cpp'), 'r') as f:
#         code = f.read()
#     solution_str = f"""abc```cpp
# {code}
# ```lalala"""  

    try:
        baseline_data, compile_success = prepare_environment(solution_str, pyvrp_id)
    except Exception as e:
        print(f"Error while compile: {e}")
        return penalty_compile_fail
    if compile_success == False:
        print("Compile failed.")
        return penalty_compile_fail

    import subprocess
    from pathlib import Path

    def run_benchmark_command(path, pyvrp_id, test_all, seed):
        # for debug
        # Build complete command parameters
        cmd = [
            "python", "bench.py", 
            "--pyvrp_id", pyvrp_id,
            "--random_seed", str(seed)
        ]
        
        # Add test_all parameter based on conditions
        if pyvrp_id == "baseline" or test_all:
            cmd.append("--test_all")
        
        # First subcommand: change directory
        benchmark_dir = str(path / "benchmark")

        print("benchmark_dir", benchmark_dir)
        print("cmd", cmd)
        
        # Second subcommand: run python script
        result = subprocess.run(
            cmd,
            cwd=benchmark_dir,  # Set working directory
            capture_output=True,  # Capture output
            text=True
        )

        # Check return code
        if result.returncode == 0:
            print("Command executed successfully!")
        else:
            print(f"Command execution failed, return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
        
        return result.returncode
    exit_code_2 = os.system("cd " + str(path / "benchmark") + " && python bench.py --pyvrp_id " + pyvrp_id + (" --test_all" if pyvrp_id == "baseline" or test_all else "") + " --random_seed " + str(seed) + " --problem_type " + problem_type + " 2>/dev/null")
    # exit_code_2 = run_benchmark_command(path, pyvrp_id, test_all, seed) # better log for debug

    if exit_code_2 != 0:
        return penalty_runtime_error


    # Skip AST check for validation/test to improve performance
    # from verl.workers.reward_manager.cpp_duplicate import is_duplicate_cpp_code
    # Only perform AST check during training phase
    is_training = phase == "train"
    
    if USE_AST_CHECK and is_training:
        try:
            is_duplicate = is_duplicate_cpp_code(
                solution_str.split("```cpp")[1].split("```")[0],
                parent_code, 
                include_dirs=[
                    '../pyvrp/pyvrp/cpp',
                    '../pyvrp/pyvrp/cpp/crossover'
                ]
            )
            print(f"is_duplicate: {is_duplicate}")
        except Exception as e:
            print(f"Error while check duplicate: {e}")
            is_duplicate = False
    else:
        is_duplicate = False

    if is_duplicate:
        print("Duplicate code.")
        return -0.9

    try:
        results_path = glob(str(path / "benchmark" / f"pyvrp_{pyvrp_id}" /"results.json"))[0]
        with open(results_path, "r") as f:
            results = json.load(f)
    except Exception as e:
        return penalty_runtime_error
    

    code = solution_str.split("```cpp")[1].split("```")[0]

    baseline_avg_score = 0
    score = 0
    try:
        scaling_factor = {
            "CVRP": 1,
            "CVRPTW": 20
        }[problem_type]
    except Exception as e:
        raise ValueError(f"Invalid problem type: {problem_type}. Set PROBLEM_TYPE environment variable!")
    for k in results['all_costs']:
        instance_name = k['instance_name']
        gap = k['gap']
        cost = k['cost']
        baseline_avg_score += baseline_data[instance_name]
        if problem_type == "CVRP":
            score += gap
        elif problem_type == "CVRPTW":
            score += cost
    baseline_avg_score /= len(results['all_costs'])
    score /= len(results['all_costs'])
    
    if USE_SMOOTH_REWARD:
        # Use smooth reward calculation
        reward = max((baseline_avg_score - score) * scaling_factor / baseline_avg_score, score_relative_lowerbound)
    else:
        # Use 0-1 binary reward: return 1 if score is less than baseline, otherwise return 0
        reward = 1.0 if score < baseline_avg_score else 0.0
    
    if reward > 0:
        add_new_individual(code, score)
    return reward

def eval_baseline():
    from verl.utils.vrp_baseline_code import BASELINE_CODE
    # from verl.utils.vrp_gpt_code import BASELINE_CODE
    code = "```cpp" + BASELINE_CODE + "```"
    compute_score(code, None, "baseline", -1, 0, test_all=True, parent_code=None, phase="test")

    # benchmark/pyvrp_baseline/result.json -> benchmark/baseline_data.json
    results_path = glob(str(path / "benchmark" / "pyvrp_baseline" / "results.json"))[0]
    with open(results_path, "r") as f:
        results = json.load(f)
    try:
        with open(os.path.join(path, 'benchmark/baseline_data.json'), 'w') as f:
            json.dump(results, f)

    except:
        os._exit(1)
    return None


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--eval_baseline", action="store_true", help="Evaluate baseline score")
    args = args.parse_args()
    if args.eval_baseline:
        eval_baseline()



    