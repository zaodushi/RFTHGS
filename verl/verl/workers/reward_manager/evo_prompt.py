from verl.utils.vrp_baseline_code import BASELINE_CODE
import os

module_to_modify = os.getenv("module_to_modify", "crossover")
baseline_code = BASELINE_CODE

def evo_prompt(code:str = None):
    if code is None:
        code = baseline_code

    if module_to_modify == "crossover":
        question = f"""
# ROLE: Expert C++ Optimization Engineer for Vehicle Routing Problems

You are a senior C++ optimization engineer with expertise in algorithmic optimization, particularly for Vehicle Routing Problems (VRP). Your task is to analyze and improve the selective_route_exchange.cpp file's crossover algorithm.

## TASK OVERVIEW
You are given the file selective_route_exchange.cpp (full listing below).
Your goal is to make multiple reliable modifications that tends to create
children with better penalised cost (Solution quality ↑) while keeping runtime
and interface intact.

## THINKING PROCESS REQUIREMENTS
1. First, thoroughly analyze the current implementation to understand:
   - The algorithm's purpose and workflow
   - Key decision points and heuristics
   - Performance bottlenecks or optimization opportunities
   - Any constraints that must be preserved

2. Generate multiple algorithm optimization ideas, evaluating each on:
   - Novel heuristics or optimization strategies that can improve solution quality
   - Computational overhead and runtime efficiency
   - Integration and synergy with existing heuristic mechanisms
   - Robustness and risk of introducing bugs

3. Select the most promising ideas and implement them in the code. For your chosen ideas:
   - Justify why each is likely to improve solution quality
   - Verify they maintain the function signature and behavior
   - Double-check for compatibility with the rest of the codebase and with each other
   - Consider edge cases and verify robustness

############################################################
## HARD RULES (Mandatory Verification Checklist)

1. □ Keep the function signature and namespace exactly the same:
   pyvrp::crossover::selectiveRouteExchange(...)
   
2. □ The file must still compile under C++17 with the current #include lines.
   You may NOT remove #include directives.
   
3. □ Do not change any public headers, class interfaces, or external behaviour
   except for the improved offspring quality.
   
4. □ DO NOT fabricate or use non-existent or unmentioned attributes or methods.
   Verify every method you use exists in the provided code or documentation.
   
5. □ Wrap the code you output with ```cpp and ```.

6. □ Mark ALL your modifications with clear "// MODIFY: XXX" comments explaining the change.

############################################################
## DELIVERABLES (strict):

A. ≤ 2-sentence summary of the optimization idea, clearly explaining how it improves solution quality.

B. Output the FULL C++ code with your modifications. Mark all changes with "// MODIFY: XXX" comments.

############################################################
## SCORING AND EVALUATION

We will benchmark on a fixed random seed over several CVRP instances.
Your patch should reduce the average optimal gap in most of the instances without
increasing time complexity significantly.

############################################################
    
## selective_route_exchange.cpp
```cpp
{code}
```

## Extra Information:
## DOMAIN KNOWLEDGE: CVRP AND CROSSOVER OPERATIONS

The Selective Route Exchange is a crossover operation for the Capacitated Vehicle Routing Problem (CVRP). The algorithm:
1. Selects routes from two parent solutions
2. Exchanges these routes to create offspring
3. Aims to preserve beneficial route structures while creating new combinations

## Essential Fields and Methods for CVRP Crossover

**ProblemData Key Methods:**
- `centroid()` - Returns `std::pair<double, double> const &` center of all client locations
- `numLocations()` - Returns `size_t` total number of locations (depots + clients)
- `numClients()` - Returns `size_t` number of client locations
- `location(idx)` - Returns `Location` union that implicitly converts to `Client const &` or `Depot const &`, providing access to client/depot data with coordinates (x, y)

**Route Key Methods:**
- `centroid()` - Returns `std::pair<double, double> const &` center of route's client locations
- `vehicleType()` - Returns `size_t` vehicle type index
- `begin()` / `end()` - Iterator support for visiting clients in route
- `size()` - Returns `size_t` number of clients in route
- `visits()` - Returns `Route::Visits` (vector-like) all client indices in route order
    """
    elif module_to_modify == "subpopulation":
        question = f"""
# ROLE: Expert C++ Optimization Engineer for Vehicle Routing Problems

You are a senior C++ optimization engineer with expertise in algorithmic optimization, particularly for Vehicle Routing Problems (VRP). Your task is to analyze and improve the subpopulation algorithm. 

## TASK OVERVIEW
You are given the file SubPopulation.cpp (full listing below).
Your goal is to improve the subpopulation algorithm by adding a new index to measure the individuals' quality while keeping interface intact.

############################################################
## HARD RULES (Mandatory Verification Checklist)

1. □ Keep the function signature and namespace exactly the same. You may define new methods.
   
2. □ The file must still compile under C++17 with the current #include lines.
   You may NOT remove #include directives.
   
3. □ Do not change any public headers, class interfaces, or external behaviour
   except for the improved solution quality.
   
4. □ DO NOT fabricate or use non-existent or unmentioned attributes or methods.
   Verify every method you use exists in the provided code or documentation.
   
5. □ Wrap the code you output with ```cpp and ```.

6. □ Mark ALL your modifications with clear "// MODIFY: XXX" comments explaining the change.

############################################################
## DELIVERABLES (strict):

A. ≤ 2-sentence summary of the optimization idea, clearly explaining how it improves solution quality.

B. Output the FULL C++ code with your modifications. Mark all changes with "// MODIFY: XXX" comments.

############################################################
## SCORING AND EVALUATION

We will benchmark on a fixed random seed over several CVRP instances.
Your patch should reduce the average optimal gap in most of the instances without
increasing time complexity significantly.

############################################################

## Extra Information:
## DOMAIN KNOWLEDGE: SUBPOPULATION ALGORITHM AND GENETIC OPERATORS

The SubPopulation class manages a collection of VRP solutions with automatic survivor selection. It maintains solution diversity while preserving high-quality individuals.

### Key Algorithm Components:
1. Population management with automatic purging when exceeding maxPopSize
2. Biased fitness calculation combining cost rank and diversity rank
3. Diversity management using proximity lists and distance measures

## Essential Fields and Methods for SubPopulation

**PopulationParams Key Fields:**
- `minPopSize` - minimum population size after purging
- `generationSize` - number of solutions added between purges  
- `numElite` - number of elite solutions always preserved
- `numClose` - number of close neighbors for diversity calculation
- `maxPopSize()` - returns minPopSize + generationSize

**SubPopulation::Item Key Methods:**
- `solution` - shared_ptr to const Solution
- `fitness` - biased fitness score (lower = better)
- `proximity` - vector of pairs<double, Solution const*> sorted by distance
- `avgDistanceClosest()` - average distance to numClose nearest solutions

**CostEvaluator Key Methods:**
- `penalisedCost(*solution)` - returns Cost with penalties for infeasibility
- `cost(*solution)` - returns Cost (huge value if infeasible)

**Solution Key Methods:**
- `empty()` - returns bool if solution has no routes
- `isFeasible()` - returns bool for constraint satisfaction
- `numRoutes()` - returns size_t number of routes
- `distance()` - returns Distance total route distance
- `excessLoad()` - returns vector<Load> capacity violations
- `timeWarp()` - returns Duration time window violations

**Algorithm Flow:**
- Solutions added via `add()` trigger purging when population exceeds maxPopSize
- Purging: first removes duplicates, then removes high-fitness solutions
- Fitness calculation: `fitness = (costRank + divWeight * divRank) / (2 * popSize)`
- Diversity measured via `divOp(*solution1, *solution2)` (higher = more diverse)
    
## SubPopulation.cpp
```cpp
{code}
```
"""
    elif module_to_modify == "subpopulation_new_prompt":
      question = f"""
# ROLE: Expert C++ Optimization Engineer for Vehicle Routing Problems

You are a senior C++ optimization engineer with expertise in algorithmic optimization, particularly for Vehicle Routing Problems (VRP). Your task is to analyze and improve the subpopulation algorithm. 

## TASK OVERVIEW
You are given the file SubPopulation.cpp (full listing below).
Your goal is to improve the subpopulation algorithm to enhance solution quality and selection effectiveness while keeping interface intact. You may optimize fitness calculation, diversity measurement, selection logic, or other algorithmic components to achieve better filtering and screening results.

############################################################
## HARD RULES (Mandatory Verification Checklist)

1. □ Keep the function signature and namespace exactly the same. You may define new methods.
   
2. □ The file must still compile under C++17 with the current #include lines.
   You may NOT remove #include directives.
   
3. □ Do not change any public headers, class interfaces, or external behaviour
   except for the improved solution quality.
   
4. □ DO NOT fabricate or use non-existent or unmentioned attributes or methods.
   Verify every method you use exists in the provided code or documentation.
   
5. □ Wrap the code you output with ```cpp and ```.

6. □ Mark ALL your modifications with clear "// MODIFY: XXX" comments explaining the change.

############################################################
## DELIVERABLES (strict):

A. ≤ 2-sentence summary of the optimization approach, clearly explaining how it improves selection effectiveness and solution quality.

B. Output the FULL C++ code with your modifications. Mark all changes with "// MODIFY: XXX" comments.

############################################################
## SCORING AND EVALUATION

We will benchmark on a fixed random seed over several CVRP instances.
Your patch should reduce the average optimal gap in most of the instances without
increasing time complexity significantly.

############################################################

## Extra Information:
## DOMAIN KNOWLEDGE: SUBPOPULATION ALGORITHM AND GENETIC OPERATORS

The SubPopulation class manages a collection of VRP solutions with automatic survivor selection. It maintains solution diversity while preserving high-quality individuals.

### Key Algorithm Components:
1. Population management with automatic purging when exceeding maxPopSize
2. Biased fitness calculation combining cost rank and diversity rank
3. Diversity management using proximity lists and distance measures

## Essential Fields and Methods for SubPopulation

**PopulationParams Key Fields:**
- `minPopSize` - minimum population size after purging
- `generationSize` - number of solutions added between purges  
- `numElite` - number of elite solutions always preserved
- `numClose` - number of close neighbors for diversity calculation
- `maxPopSize()` - returns minPopSize + generationSize

**SubPopulation::Item Key Methods:**
- `solution` - shared_ptr to const Solution
- `fitness` - biased fitness score (lower = better)
- `proximity` - vector of pairs<double, Solution const*> sorted by distance
- `avgDistanceClosest()` - average distance to numClose nearest solutions

**CostEvaluator Key Methods:**
- `penalisedCost(*solution)` - returns Cost with penalties for infeasibility
- `cost(*solution)` - returns Cost (huge value if infeasible)

**Solution Key Methods:**
- `empty()` - returns bool if solution has no routes
- `isFeasible()` - returns bool for constraint satisfaction
- `numRoutes()` - returns size_t number of routes
- `distance()` - returns Distance total route distance
- `excessLoad()` - returns vector<Load> capacity violations
- `timeWarp()` - returns Duration time window violations

**Algorithm Flow:**
- Solutions added via `add()` trigger purging when population exceeds maxPopSize
- Purging: first removes duplicates, then removes high-fitness solutions
- Fitness calculation: `fitness = (costRank + divWeight * divRank) / (2 * popSize)`
- Diversity measured via `divOp(*solution1, *solution2)` (higher = more diverse)
    
## SubPopulation.cpp
```cpp
{code}
```
"""
    return question