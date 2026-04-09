import os
import json
import glob
import shutil
import subprocess
from pathlib import Path
import sys

ROOT_PATH = Path(__file__).parent
FIXED_BASELINE_SCORES = {'X-n1001-k43': 4.639624075737682, 'X-n101-k25': 0.0, 'X-n106-k14': 0.6448676124724984, 'X-n110-k13': 0.23378531828201188, 'X-n115-k10': 0.0, 'X-n120-k6': 0.21002100210021002, 'X-n125-k30': 0.4231260915752894, 'X-n129-k18': 1.4685556323427782, 'X-n134-k13': 1.117625503847563, 'X-n139-k10': 0.27225901398086827, 'X-n143-k7': 0.6687898089171975, 'X-n148-k46': 0.8815135334192598, 'X-n153-k22': 0.8765315739868049, 'X-n157-k13': 0.42071580943351505, 'X-n162-k11': 0.2121940868581129, 'X-n167-k10': 1.0166853140049619, 'X-n172-k51': 0.37494244304602364, 'X-n176-k26': 1.2946540617418223, 'X-n181-k23': 0.7782862059525206, 'X-n186-k15': 0.5632636156554153, 'X-n190-k8': 1.0895170789163722, 'X-n195-k51': 0.5313736574335783, 'X-n200-k36': 0.6913858445150056, 'X-n204-k19': 1.0222335803731153, 'X-n209-k16': 2.299712943632568, 'X-n214-k11': 2.47789240972734, 'X-n219-k73': 0.16582337684425358, 'X-n223-k34': 2.6436184682345374, 'X-n228-k23': 1.359645715173646, 'X-n233-k16': 0.9048361934477379, 'X-n237-k14': 2.100436358257525, 'X-n242-k48': 1.7582869089195299, 'X-n247-k50': 1.0624027472232656, 'X-n251-k28': 2.507496639437494, 'X-n256-k16': 0.9926216890493125, 'X-n261-k13': 2.116123202048347, 'X-n266-k58': 1.9661358276583905, 'X-n270-k35': 0.9407497662293502, 'X-n275-k28': 1.5580136502706519, 'X-n280-k17': 1.8774438110019998, 'X-n284-k15': 2.739048749134777, 'X-n289-k60': 1.4650397788777838, 'X-n294-k50': 1.3824982506732257, 'X-n298-k31': 3.1141363091934213, 'X-n303-k21': 1.6700404858299596, 'X-n308-k13': 2.165590316717584, 'X-n313-k71': 1.390853120381102, 'X-n317-k53': 0.3445855401697403, 'X-n322-k28': 1.4882348997787758, 'X-n327-k20': 4.547435711172454, 'X-n331-k15': 2.8744132210147257, 'X-n336-k84': 1.4290746238615206, 'X-n344-k43': 1.6052318668252081, 'X-n351-k40': 2.467562557924004, 'X-n359-k29': 2.898747694398602, 'X-n367-k17': 2.792145174015955, 'X-n376-k94': 0.4373345609391184, 'X-n384-k52': 1.7288444040036397, 'X-n393-k38': 3.285415577626764, 'X-n401-k29': 1.5902288599328838, 'X-n411-k19': 1.3443587662337662, 'X-n420-k130': 1.795951687415351, 'X-n429-k61': 1.928218918547266, 'X-n439-k37': 1.6487593086202634, 'X-n449-k29': 2.8135353864537507, 'X-n459-k26': 2.241186461742408, 'X-n469-k138': 1.4407818811309867, 'X-n480-k70': 2.5489385012688794, 'X-n491-k59': 2.6984341861829337, 'X-n502-k39': 0.7901655447375264, 'X-n513-k21': 2.627990578901698, 'X-n524-k153': 0.6701467724929331, 'X-n536-k96': 1.9526390148240307, 'X-n548-k50': 1.9111880046136103, 'X-n561-k42': 2.750661329213194, 'X-n573-k30': 2.4194344128036627, 'X-n586-k159': 1.3451312553857795, 'X-n599-k92': 2.498824353855658, 'X-n613-k62': 3.5911648610061313, 'X-n627-k43': 1.9480728395856122, 'X-n641-k35': 3.104390427736951, 'X-n655-k131': 0.6677280389586064, 'X-n670-k130': 0.898641445480141, 'X-n685-k75': 2.9132761527747233, 'X-n701-k44': 2.935683507684045, 'X-n716-k35': 2.709058630945519, 'X-n733-k159': 2.9731178453156324, 'X-n749-k98': 3.2962766439322366, 'X-n766-k71': 2.4270868839420716, 'X-n783-k48': 3.9869588041886552, 'X-n801-k40': 4.067602406187339, 'X-n819-k171': 1.6563264841482157, 'X-n837-k142': 2.3599002771798885, 'X-n856-k95': 1.6905524644523127, 'X-n876-k59': 2.6072770118530904, 'X-n895-k37': 3.5499443000371333, 'X-n916-k207': 1.5456636055155402, 'X-n936-k151': 2.097728214595185, 'X-n957-k87': 2.114315801790206, 'X-n979-k58': 1.507026627218935}

def extract_cpp_code(solution_str):
    """Extract C++ code from an LLM response string."""
    try:
        print(solution_str)
        code = solution_str.split("```cpp")[-1].split("```")[0]
        return code
    except Exception as e:
        print(f"Failed to extract C++ code: {e}")
        return None

def prepare_test_environment(code_str, test_id="test", crossover_type="mtsp", module_to_modify="crossover"):
    """Set up an isolated test environment for evaluating LLM-generated code."""
    test_dir = ROOT_PATH / "benchmark" / f"pyvrp_{test_id}"

    if test_dir.exists():
        shutil.rmtree(test_dir)

    test_dir.mkdir(parents=True, exist_ok=True)

    src_pyvrp = ROOT_PATH / "pyvrp"
    dst_pyvrp = test_dir / "pyvrp"

    try:
        shutil.copytree(src_pyvrp, dst_pyvrp,
                        ignore=shutil.ignore_patterns('.venv', 'docs', 'tests', 'examples'))
    except Exception as e:
        print(f"Failed to copy pyvrp: {e}")
        return False

    if module_to_modify in ("subpopulation", "subpopulation_new_prompt"):
        target_filename = "SubPopulation.cpp"
    elif crossover_type == "tsp":
        target_filename = "ordered_crossover.cpp"
    elif crossover_type == "mtsp":
        target_filename = "selective_route_exchange.cpp"
    else:
        raise ValueError(f"Unknown module to modify: {module_to_modify}")

    if module_to_modify == "crossover":
        cpp_files = list(dst_pyvrp.glob(f"**/cpp/crossover/{target_filename}"))
    elif module_to_modify in ("subpopulation", "subpopulation_new_prompt"):
        cpp_files = list(dst_pyvrp.glob(f"**/cpp/{target_filename}"))
    else:
        raise ValueError(f"Unknown module to modify: {module_to_modify}")

    if not cpp_files:
        print(f"{target_filename} not found in copied pyvrp tree")
        return False

    cpp_path = cpp_files[0]
    try:
        with open(cpp_path, 'w', encoding='utf-8') as f:
            f.write(code_str)
        print(f"✅ Replaced {target_filename}")
        return True
    except Exception as e:
        print(f"Failed to write C++ code: {e}")
        return False

def compile_code(test_id="test"):
    """Compile the modified pyvrp code and move the resulting .so files into place."""
    test_dir = ROOT_PATH / "benchmark" / f"pyvrp_{test_id}" / "pyvrp"

    try:
        cmd = f"cd {test_dir} && meson compile -C build"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return False

        build_dir = test_dir / "build"
        so_files = list(build_dir.glob("_*.so"))

        if not so_files:
            print("No .so files produced by the build")
            return False

        for so_file in so_files:
            filename = so_file.name
            if filename.startswith("_pyvrp"):
                dest_dir = test_dir / "pyvrp"
            elif filename.startswith("_crossover"):
                dest_dir = test_dir / "pyvrp" / "crossover"
            elif filename.startswith("_diversity"):
                dest_dir = test_dir / "pyvrp" / "diversity"
            elif filename.startswith("_repair"):
                dest_dir = test_dir / "pyvrp" / "repair"
            elif filename.startswith("_search"):
                dest_dir = test_dir / "pyvrp" / "search"
            else:
                continue
                
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(so_file), str(dest_dir / filename))

        return True
    except Exception as e:
        print(f"Error during compilation or file move: {e}")
        return False

def run_benchmark(test_id="test", random_seed=42, iters=800, num_procs=16):
    """Run the PyVRP benchmark and return parsed results."""
    cmd = [
        "python", "-u",
        str(ROOT_PATH / "benchmark" / "bench.py"),
        "--pyvrp_id", test_id,
        "--random_seed", str(random_seed),
        "--iters", str(iters),
        "--test_all",
        "--problem_type", "CVRP_all"
        ]
    try:
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1) as p:
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            p.wait()
        if p.returncode != 0:
            print(f"Benchmark run failed: {p.stderr}")
            return None

        results_file = ROOT_PATH / "benchmark" / f"pyvrp_{test_id}" / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None

def extract_instance_details(datasets):
    """Extract per-instance details from dataset results."""
    instance_details = {}

    for dataset_name, dataset_info in datasets.items():
        # Parse dataset name, e.g. "cvrp/50" -> vrp_type="cvrp", size="50"
        parts = dataset_name.split("/")
        if len(parts) >= 2:
            vrp_type = parts[0]
            size = parts[1]
        else:
            vrp_type = "unknown"
            size = "unknown"
        
        if vrp_type not in instance_details:
            instance_details[vrp_type] = {}
            
        if size not in instance_details[vrp_type]:
            instance_details[vrp_type][size] = {
                "average_cost": dataset_info["average_cost"],
                "num_instances": dataset_info["num_instances"],
                "problem_size": dataset_info["problem_size"],
                "time_seconds": dataset_info["time_seconds"],
                "instance_costs": dataset_info.get("instance_costs", [])
            }
    
    return instance_details

def calculate_score(results_dict):
    """Compute the average gap and improvement over the fixed baseline."""
    if not results_dict:
        return -1

    baseline_scores = []
    actual_scores = []

    for instance, actual_gap in results_dict.items():
        baseline_scores.append(FIXED_BASELINE_SCORES[instance])
        actual_scores.append(actual_gap)

    try:
        avg_gap = sum(actual_scores) / len(actual_scores)
        baseline_avg_gap = sum(baseline_scores) / len(baseline_scores)
        gap_diff = baseline_avg_gap - avg_gap
        improvement = gap_diff / baseline_avg_gap
        return {
            "avg_gap": avg_gap,
            "baseline_avg_gap": baseline_avg_gap,
            "gap_diff": gap_diff,
            "improvement": improvement
        }
    except:
        return -1

def run_baseline_test(random_seed=42, iters=800, num_procs=16):
    """Run baseline test using the default MTSP crossover (legacy interface)."""
    return run_baseline_test_with_crossover(random_seed, iters, num_procs, "mtsp")

def run_baseline_test_with_crossover(random_seed=42, iters=800, num_procs=16, crossover_type="mtsp"):
    """Run the baseline code and return benchmark results."""
    from vllm_evaluate import get_baseline_code

    baseline_code = get_baseline_code(crossover_type)
    baseline_test_id = f"baseline_{crossover_type}"

    if crossover_type == "tsp":
        print("Running baseline test (TSP - ordered_crossover)...")
    else:
        print("Running baseline test (MTSP - selective_route_exchange)...")

    if not prepare_test_environment(baseline_code, baseline_test_id, crossover_type, "crossover"):
        print("❌ Baseline test environment setup failed")
        return None

    if not compile_code(baseline_test_id):
        print("❌ Baseline compilation failed")
        return None

    baseline_results = run_benchmark(baseline_test_id, random_seed, iters, num_procs)

    if baseline_results is None:
        print("❌ Baseline benchmark run failed")
        return None

    print("✅ Baseline test complete")
    return baseline_results

def calculate_gaps(llm_results, baseline_results):
    """Calculate per-dataset gap between LLM results and the baseline."""
    gaps = {}
    total_gap = 0.0
    count = 0

    print("📊 Gap relative to baseline:")

    for dataset_name in llm_results['datasets']:
        if dataset_name in baseline_results['datasets']:
            llm_cost = llm_results['datasets'][dataset_name]['average_cost']
            baseline_cost = baseline_results['datasets'][dataset_name]['average_cost']

            gap = (llm_cost - baseline_cost) / baseline_cost
            gaps[dataset_name] = gap
            total_gap += gap
            count += 1

            print(f"   {dataset_name}: LLM={llm_cost:.4f}, Baseline={baseline_cost:.4f}, Gap={gap:.4f}")

    average_gap = total_gap / count if count > 0 else 0.0
    print(f"📈 Average gap: {average_gap:.4f}")

    return gaps, average_gap

def evaluate_llm_code_with_baseline(code, baseline_results, test_id="test", random_seed=42, problem_types="['CVRP']", iters=800, num_procs=16, crossover_type="mtsp", module_to_modify="crossover"):
    """Evaluate LLM-generated C++ code against pre-computed baseline results."""
    import ast
    problem_types = ast.literal_eval(problem_types)
    print(f"Problem types: {problem_types}")
    if module_to_modify == "crossover":
        print(f"Crossover type: {crossover_type}")
    else:
        print(f"Module: {module_to_modify}")
    print(f"Code length: {len(code)} chars")

    if baseline_results is None:
        print("❌ Baseline results are empty")
        return -1.0

    print("✅ Using cached baseline results")

    if not prepare_test_environment(code, test_id, crossover_type, module_to_modify):
        print("❌ LLM code test environment setup failed")
        return -1.0

    print("✅ Test environment ready")

    if not compile_code(test_id):
        print("❌ LLM code compilation failed")
        return -1.0

    print("✅ Compilation successful")

    llm_results = run_benchmark(test_id, random_seed, iters, num_procs)

    if llm_results is None:
        print("❌ LLM benchmark run failed")
        return -1.0

    print("✅ LLM benchmark complete")

    try:
        gaps, average_gap = calculate_gaps(llm_results, baseline_results)
        
        detailed_results = {
            "average_gap": average_gap,
            "llm_average_cost": llm_results['summary']['average_cost_across_all_datasets'],
            "baseline_average_cost": baseline_results['summary']['average_cost_across_all_datasets'],
            "gaps": gaps,
            "llm_datasets": llm_results['datasets'],
            "baseline_datasets": baseline_results['datasets'],
            "instance_details": extract_instance_details(llm_results['datasets'])
        }
        return detailed_results
        
    except Exception as e:
        print(f"❌ Failed to process results: {e}")
        import traceback
        traceback.print_exc()
        return -1.0

def evaluate_llm_code(code, test_id="test", random_seed=42, problem_types="['CVRP']", iters=800, num_procs=16):
    """Evaluate LLM-generated C++ code (legacy interface; re-runs baseline each call)."""
    import ast
    problem_types = ast.literal_eval(problem_types)
    print(f"Problem types: {problem_types}")
    print(f"Code length: {len(code)} chars")

    baseline_results = run_baseline_test(random_seed, iters, num_procs)
    if baseline_results is None:
        print("❌ Baseline test failed")
        return -1.0

    return evaluate_llm_code_with_baseline(code, baseline_results, test_id, random_seed, problem_types, iters, num_procs)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM-generated C++ code")
    parser.add_argument("--code", type=str, help="LLM-generated code string or path to a file")
    parser.add_argument("--test_id", type=str, default="test", help="test identifier")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--problem_types", type=str, default="['cvrp']", help="problem types list")
    parser.add_argument("--iters", type=int, default=800, help="max HGS iterations")
    parser.add_argument("--num_procs", type=int, default=16, help="parallel worker processes")

    args = parser.parse_args()

    if args.code and os.path.isfile(args.code):
        with open(args.code, 'r', encoding='utf-8') as f:
            code_str = f.read()
    else:
        code_str = args.code or """
```cpp
#include "crossover.h"
// ... C++ code here ...
```
        """

    score = evaluate_llm_code(code_str, args.test_id, args.seed, args.problem_types, args.iters, args.num_procs)
    print(f"\nFinal evaluation result: {score}")

if __name__ == "__main__":
    # Example: python vllm_evaluate.py <safetensor_path> -n 16 --problem_types "['cvrp']" --iters 800 2>&1 | tee eval.log
    main()