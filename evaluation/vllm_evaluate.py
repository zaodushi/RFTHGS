import argparse
import json
import numpy as np
import os
import sys
from typing import List, Dict

from vllm import LLM, SamplingParams
from .evaluate_llm_code import evaluate_llm_code

# LLM configuration parameters
LLM_CONFIG = {
    'gpu_memory_utilization': 0.8,
    'max_model_len': 16384,
    'swap_space': 4,
    'tensor_parallel_size': 1,
    'dtype': 'auto',
    'trust_remote_code': True,
}

# Sampling parameter configuration
SAMPLING_CONFIG = {
    'temperature': 1,
    'top_p': 1,
    'top_k': 100,
    'max_tokens': 16384,
}

# Baseline code template - MTSP (Selective Route Exchange)
BASELINE_CODE_MTSP = """
#include "selective_route_exchange.h"

#include "DynamicBitset.h"

#include <cmath>
#include <vector>

using Client = size_t;
using Clients = std::vector<Client>;
using Route = pyvrp::Route;
using Routes = std::vector<Route>;

namespace
{
// Angle of the given route w.r.t. the centroid of all client locations.
double routeAngle(pyvrp::ProblemData const &data, Route const &route)
{
    auto const [dataX, dataY] = data.centroid();
    auto const [routeX, routeY] = route.centroid();
    return std::atan2(routeY - dataY, routeX - dataX);
}

Routes sortByAscAngle(pyvrp::ProblemData const &data, Routes routes)
{
    auto cmp = [&data](Route const &a, Route const &b)
    { return routeAngle(data, a) < routeAngle(data, b); };

    std::sort(routes.begin(), routes.end(), cmp);
    return routes;
}
}  // namespace

pyvrp::Solution pyvrp::crossover::selectiveRouteExchange(
    std::pair<Solution const *, Solution const *> const &parents,
    ProblemData const &data,
    CostEvaluator const &costEvaluator,
    std::pair<size_t, size_t> const &startIndices,
    size_t const numMovedRoutes)
{
    // We create two candidate offsprings, both based on parent A:
    // Let A and B denote the set of customers selected from parents A and B
    // Ac and Bc denote the complements: the customers not selected
    // Let v denote union and ^ intersection
    // Parent A: A v Ac
    // Parent B: B v Bc

    // Offspring 1:
    // B and Ac\B, remainder A\B unplanned
    // (note B v (Ac\B) v (A\B) = B v ((Ac v A)\B) = B v Bc = all)
    // Note Ac\B = (A v B)c

    // Offspring 2:
    // A^B and Ac, remainder A\B unplanned
    // (note A^B v Ac v A\B = (A^B v A\B) v Ac = A v Ac = all)

    auto startA = startIndices.first;
    auto startB = startIndices.second;

    size_t nRoutesA = parents.first->numRoutes();
    size_t nRoutesB = parents.second->numRoutes();

    if (startA >= nRoutesA)
        throw std::invalid_argument("Expected startA < nRoutesA.");

    if (startB >= nRoutesB)
        throw std::invalid_argument("Expected startB < nRoutesB.");

    if (numMovedRoutes < 1 || numMovedRoutes > std::min(nRoutesA, nRoutesB))
    {
        auto msg = "Expected numMovedRoutes in [1, min(nRoutesA, nRoutesB)]";
        throw std::invalid_argument(msg);
    }

    // Sort parents' routes by (ascending) polar angle.
    auto const routesA = sortByAscAngle(data, parents.first->routes());
    auto const routesB = sortByAscAngle(data, parents.second->routes());

    DynamicBitset selectedA(data.numLocations());
    DynamicBitset selectedB(data.numLocations());

    // Routes are sorted on polar angle, so selecting adjacent routes in both
    // parents should result in a large overlap when the start indices are
    // close to each other.
    for (size_t r = 0; r < numMovedRoutes; r++)
    {
        for (Client c : routesA[(startA + r) % nRoutesA])
            selectedA[c] = true;

        for (Client c : routesB[(startB + r) % nRoutesB])
            selectedB[c] = true;
    }

    // For the selection, we want to minimize |A\B| as these need replanning
    while (true)
    {
        // Difference for moving 'left' in parent A
        int differenceALeft = 0;

        for (Client c : routesA[(startA - 1 + nRoutesA) % nRoutesA])
            differenceALeft += !selectedB[c];

        for (Client c : routesA[(startA + numMovedRoutes - 1) % nRoutesA])
            differenceALeft -= !selectedB[c];

        // Difference for moving 'right' in parent A
        int differenceARight = 0;

        for (Client c : routesA[(startA + numMovedRoutes) % nRoutesA])
            differenceARight += !selectedB[c];

        for (Client c : routesA[startA])
            differenceARight -= !selectedB[c];

        // Difference for moving 'left' in parent B
        int differenceBLeft = 0;

        for (Client c : routesB[(startB - 1 + numMovedRoutes) % nRoutesB])
            differenceBLeft += selectedA[c];

        for (Client c : routesB[(startB - 1 + nRoutesB) % nRoutesB])
            differenceBLeft -= selectedA[c];

        // Difference for moving 'right' in parent B
        int differenceBRight = 0;

        for (Client c : routesB[startB])
            differenceBRight += selectedA[c];

        for (Client c : routesB[(startB + numMovedRoutes) % nRoutesB])
            differenceBRight -= selectedA[c];

        int const bestDifference = std::min({differenceALeft,
                                             differenceARight,
                                             differenceBLeft,
                                             differenceBRight});

        if (bestDifference >= 0)  // there are no further improving moves
            break;

        if (bestDifference == differenceALeft)
        {
            for (Client c : routesA[(startA + numMovedRoutes - 1) % nRoutesA])
                selectedA[c] = false;

            startA = (startA - 1 + nRoutesA) % nRoutesA;
            for (Client c : routesA[startA])
                selectedA[c] = true;
        }
        else if (bestDifference == differenceARight)
        {
            for (Client c : routesA[startA])
                selectedA[c] = false;

            startA = (startA + 1) % nRoutesA;
            for (Client c : routesA[(startA + numMovedRoutes - 1) % nRoutesA])
                selectedA[c] = true;
        }
        else if (bestDifference == differenceBLeft)
        {
            for (Client c : routesB[(startB + numMovedRoutes - 1) % nRoutesB])
                selectedB[c] = false;

            startB = (startB - 1 + nRoutesB) % nRoutesB;
            for (Client c : routesB[startB])
                selectedB[c] = true;
        }
        else if (bestDifference == differenceBRight)
        {
            for (Client c : routesB[startB])
                selectedB[c] = false;

            startB = (startB + 1) % nRoutesB;
            for (Client c : routesB[(startB + numMovedRoutes - 1) % nRoutesB])
                selectedB[c] = true;
        }
    }

    // Identify differences between route sets
    auto const selectedBNotA = selectedB & ~selectedA;

    std::vector<Clients> visits1(nRoutesA);
    std::vector<Clients> visits2(nRoutesA);

    // Replace selected routes from parent A with routes from parent B
    for (size_t r = 0; r < numMovedRoutes; r++)
    {
        size_t indexA = (startA + r) % nRoutesA;
        size_t indexB = (startB + r) % nRoutesB;

        for (Client c : routesB[indexB])
        {
            visits1[indexA].push_back(c);  // c in B

            if (!selectedBNotA[c])
                visits2[indexA].push_back(c);  // c in A^B
        }
    }

    // Move routes from parent A that are kept
    for (size_t r = numMovedRoutes; r < nRoutesA; r++)
    {
        size_t indexA = (startA + r) % nRoutesA;

        for (Client c : routesA[indexA])
        {
            if (!selectedBNotA[c])
                visits1[indexA].push_back(c);  // c in Ac\B

            visits2[indexA].push_back(c);  // c in Ac
        }
    }

    // Turn visits back into routes.
    Routes routes1;
    Routes routes2;
    for (size_t r = 0; r < nRoutesA; r++)
    {
        if (!visits1[r].empty())
            routes1.emplace_back(data, visits1[r], routesA[r].vehicleType());

        if (!visits2[r].empty())
            routes2.emplace_back(data, visits2[r], routesA[r].vehicleType());
    }

    auto const sol1 = Solution(data, routes1);
    auto const sol2 = Solution(data, routes2);

    auto const cost1 = costEvaluator.penalisedCost(sol1);
    auto const cost2 = costEvaluator.penalisedCost(sol2);
    return cost1 < cost2 ? sol1 : sol2;
}
"""

# Baseline code template - TSP (ordered crossover)
BASELINE_CODE_TSP = """
#include "ordered_crossover.h"

#include "DynamicBitset.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace
{
using Client = size_t;

// Depot value, which is never in a route (since it's not a client). We use
// this as filler to account for possibly missing clients.
static constexpr size_t UNUSED = 0;
}  // namespace

pyvrp::Solution pyvrp::crossover::orderedCrossover(
    std::pair<Solution const *, Solution const *> const &parents,
    ProblemData const &data,
    std::pair<size_t, size_t> const &indices)
{
    assert(data.numVehicles() == 1);
    assert(parents.first->numClients() > 0 && parents.second->numClients() > 0);

    auto const [start, end] = indices;
    auto const numClients = data.numClients();

    // New route. This route is initially empty, indicated by all UNUSED
    // values. Any such values that remain after crossover are filtered away.
    std::vector<Client> newRoute(numClients, UNUSED);
    DynamicBitset isInserted(data.numLocations());  // tracks inserted clients

    // Insert the clients from the first route into the new route, from start
    // to end (possibly wrapping around the end of the route).
    size_t insertIdx = start;
    auto const &route1 = parents.first->routes()[0];
    for (; insertIdx % route1.size() != end % route1.size(); ++insertIdx)
    {
        newRoute[insertIdx % numClients] = route1[insertIdx % route1.size()];
        isInserted[route1[insertIdx % route1.size()]] = true;
    }

    // Fill the route with clients from the second parent, in the order of
    // their visits in the second route.
    auto const &route2 = parents.second->routes()[0];
    for (size_t idx = 0; idx != route2.size(); ++idx)
    {
        Client const client = route2[(end + idx) % route2.size()];
        if (!isInserted[client])
        {
            newRoute[insertIdx % numClients] = client;
            insertIdx++;
        }
    }

    // Remove the UNUSED values from the new route. These were needed because
    // we cannot assume both parent solutions have all the same clients (for
    // example, solutions to instances with optional clients typically do not).
    std::vector<Client> offspring;
    std::copy_if(newRoute.begin(),
                 newRoute.end(),
                 std::back_inserter(offspring),
                 [](auto client) { return client != UNUSED; });

    return {data, {offspring}};
}
"""

BASELINE_CODE_SUBPOPULATION = """
#include "SubPopulation.h"

#include <numeric>
#include <stdexcept>

using pyvrp::PopulationParams;
using pyvrp::SubPopulation;
using const_iter = std::vector<SubPopulation::Item>::const_iterator;
using iter = std::vector<SubPopulation::Item>::iterator;

PopulationParams::PopulationParams(size_t minPopSize,
                                   size_t generationSize,
                                   size_t numElite,
                                   size_t numClose,
                                   double lbDiversity,
                                   double ubDiversity)
    : minPopSize(minPopSize),
      generationSize(generationSize),
      numElite(numElite),
      numClose(numClose),
      lbDiversity(lbDiversity),
      ubDiversity(ubDiversity)
{
    if (lbDiversity < 0 || lbDiversity > 1)
        throw std::invalid_argument("lb_diversity must be in [0, 1].");

    if (ubDiversity < 0 || ubDiversity > 1)
        throw std::invalid_argument("ub_diversity must be in [0, 1].");

    if (ubDiversity <= lbDiversity)
    {
        auto const msg = "ub_diversity <= lb_diversity not understood.";
        throw std::invalid_argument(msg);
    }
}

size_t PopulationParams::maxPopSize() const
{
    return minPopSize + generationSize;
}

SubPopulation::SubPopulation(diversity::DiversityMeasure divOp,
                             PopulationParams const &params)
    : divOp(divOp), params(params)
{
}

void SubPopulation::add(std::shared_ptr<Solution const> const &solution,
                        CostEvaluator const &costEvaluator)
{
    Item item = {&params, solution, 0.0, {}};

    for (auto &other : items_) // update distance to other solutions
    {
        auto const div = divOp(*solution, *other.solution);
        auto cmp = [](auto &elem, auto &value) { return elem.first < value; };

        auto &oProx = other.proximity;
        auto place = std::lower_bound(oProx.begin(), oProx.end(), div, cmp);
        oProx.emplace(place, div, solution.get());

        auto &iProx = item.proximity;
        place = std::lower_bound(iProx.begin(), iProx.end(), div, cmp);
        iProx.emplace(place, div, other.solution.get());
    }

    items_.push_back(item); // add solution

    if (size() > params.maxPopSize())
        purge(costEvaluator);
}

size_t SubPopulation::size() const { return items_.size(); }

SubPopulation::Item const &SubPopulation::operator[](size_t idx) const
{
    return items_[idx];
}

const_iter SubPopulation::cbegin() const { return items_.cbegin(); }

const_iter SubPopulation::cend() const { return items_.cend(); }

void SubPopulation::remove(iter const &iterator)
{
    for (auto &[params, solution, fitness, proximity] : items_)
        // Remove solution from other proximities.
        for (size_t idx = 0; idx != proximity.size(); ++idx)
            if (proximity[idx].second == iterator->solution.get())
            {
                proximity.erase(proximity.begin() + idx);
                break;
            }

    items_.erase(iterator);
}

void SubPopulation::purge(CostEvaluator const &costEvaluator)
{
    // First we remove duplicates. This does not rely on the fitness values.
    while (size() > params.minPopSize)
    {
        // Remove duplicates from the subpopulation (if they exist)
        auto const pred = [&](auto &item)
        {
            return !item.proximity.empty()
                   && *item.proximity[0].second == *item.solution;
        };

        auto const duplicate = std::find_if(items_.begin(), items_.end(), pred);
        if (duplicate == items_.end()) // there are no more duplicates
            break;

        remove(duplicate);
    }

    while (size() > params.minPopSize)
    {
        // Before using fitness, we must update fitness
        updateFitness(costEvaluator);
        auto const worstFitness = std::max_element(
            items_.begin(),
            items_.end(),
            [](auto const &a, auto const &b) { return a.fitness < b.fitness; });

        remove(worstFitness);
    }
}

void SubPopulation::updateFitness(CostEvaluator const &costEvaluator)
{
    if (items_.empty())
        return;

    /* ---------------------------------------------------------------------
     * STEP 1: rank individuals by penalised cost (identical to original)
     * ------------------------------------------------------------------ */
    // clang-format off
    std::vector<size_t> byCost(size());
    std::iota(byCost.begin(), byCost.end(), 0);
    std::stable_sort(
        byCost.begin(),
        byCost.end(),
        [&](size_t a, size_t b)
        {
            return costEvaluator.penalisedCost(*items_[a].solution)
                   < costEvaluator.penalisedCost(*items_[b].solution);
        });
    // clang-format on

    /* ---------------------------------------------------------------------
     * STEP 2: compute diversity rank (identical to original)
     * ------------------------------------------------------------------ */
    std::vector<std::pair<double, size_t>> diversity;
    for (size_t costRank = 0; costRank != size(); costRank++)
    {
        auto const dist = items_[byCost[costRank]].avgDistanceClosest();
        diversity.emplace_back(-dist, costRank); // higher is better
    }
    std::stable_sort(diversity.begin(), diversity.end());

    /* ---------------------------------------------------------------------
     * STEP 3 (NEW): rank individuals by number of routes                  *
     *              (fewer routes  -> better -> lower rank)                *
     * ------------------------------------------------------------------ */
    // MODIFY: Added route-count ranking to capture vehicle-usage quality.
    std::vector<size_t> byRoutes(size());
    std::iota(byRoutes.begin(), byRoutes.end(), 0);
    std::stable_sort(
        byRoutes.begin(),
        byRoutes.end(),
        [&](size_t a, size_t b)
        {
            return items_[a].solution->numRoutes()
                   < items_[b].solution->numRoutes();
        });

    // routeRankOfIdx[i] = rank of individual i in the route ranking.
    std::vector<size_t> routeRankOfIdx(size());
    for (size_t rank = 0; rank != byRoutes.size(); ++rank)
        routeRankOfIdx[byRoutes[rank]] = rank;

    /* ---------------------------------------------------------------------
     * STEP 4: combine the three ranks into the biased fitness
     * ------------------------------------------------------------------ */
    auto const popSize  = static_cast<double>(size());
    auto const numElite = std::min(params.numElite, size());

    auto const divWeight   = 1.0 - numElite / popSize;               // original
    auto const routeWeight = 0.5 * (1.0 - numElite / popSize);       // MODIFY: lightweight extra pressure

    for (size_t divRank = 0; divRank != size(); divRank++)
    {
        auto const costRank  = diversity[divRank].second;
        auto const idx       = byCost[costRank];
        auto const routeRank = routeRankOfIdx[idx];

        // MODIFY: fitness now blends cost, diversity and route-count.
        auto const denom =
            (1.0 /*cost*/ + divWeight + routeWeight) * popSize;

        items_[idx].fitness =
            (costRank + divWeight * divRank + routeWeight * routeRank) / denom;
    }
}

double SubPopulation::Item::avgDistanceClosest() const
{
    auto const maxSize = std::min(proximity.size(), params->numClose);
    auto result = 0.0;

    for (size_t idx = 0; idx != maxSize; ++idx)
        result += proximity[idx].first;

    return result / std::max<size_t>(maxSize, 1);
}
"""

def get_baseline_code(crossover_type="mtsp"):
    """Return the baseline C++ code for the given crossover type."""
    if crossover_type == "tsp":
        return BASELINE_CODE_TSP
    elif crossover_type == "mtsp":
        return BASELINE_CODE_MTSP
    elif crossover_type == "subpopulation":
        return BASELINE_CODE_SUBPOPULATION
    else:
        raise ValueError(f"Unknown crossover type: {crossover_type}")

class VLLMEvaluator:
    def __init__(self, model_path: str, problem_types: str, iters: int, num_procs: int = 16, crossover_type: str = "mtsp", module_to_modify: str = "crossover"):
        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            **LLM_CONFIG
        )
        self.sampling_params = SamplingParams(**SAMPLING_CONFIG)
        print("Model loaded.")
        self.problem_types = problem_types
        self.iters = iters
        self.num_procs = num_procs
        self.crossover_type = crossover_type
        self.baseline_results = None  # cached baseline results
        self.module_to_modify = module_to_modify

    def get_filename(self):
        if self.crossover_type == "tsp":
            return "ordered_crossover.cpp"
        else:
            return "selective_route_exchange.cpp"

    def create_chat_messages(self, previous_best_code: str = None) -> List[Dict[str, str]]:
        if self.module_to_modify == "crossover":
            question = f"""
# ROLE: Expert C++ Optimization Engineer for Vehicle Routing Problems

You are a senior C++ optimization engineer with expertise in algorithmic optimization, particularly for Vehicle Routing Problems (VRP). Your task is to analyze and improve the {self.get_filename()} file's crossover algorithm.

## TASK OVERVIEW
You are given the file {self.get_filename()} (full listing below).
Your goal is to make ONE small, reliable modification that tends to create
children with better penalised cost (Solution quality ↑) while keeping runtime
and interface intact.

## THINKING PROCESS REQUIREMENTS
1. First, thoroughly analyze the current implementation to understand:
   - The algorithm's purpose and workflow
   - Key decision points and heuristics
   - Performance bottlenecks or optimization opportunities
   - Any constraints that must be preserved

2. Generate at least 3 different modification approaches, evaluating each on:
   - Potential improvement to solution quality
   - Impact on runtime performance
   - Compatibility with existing code
   - Risk of introducing bugs or side effects

3. For your chosen modification:
   - Justify why it's likely to improve solution quality
   - Verify it maintains the function signature and behavior
   - Double-check for compatibility with the rest of the codebase
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

7. □ You must make at least one modification; DO NOT copy the original code.
   
8. □ Before finalizing, double-check that your modification:
   - Does not introduce new parameters
   - Does not change the function's contract
   - Is focused on improving solution quality, not runtime
   - Is fully compatible with the existing codebase
   - Uses only documented methods and attributes

############################################################
## DELIVERABLES (strict):

A. ≤ 2-sentence summary of the optimization idea, clearly explaining how it improves solution quality.

B. Output the FULL C++ code with your modifications. Mark all changes with "// MODIFY: XXX" comments.

C. Brief explanation of your verification process and why you're confident the modification will:
   - Improve solution quality
   - Maintain compatibility with the existing codebase
   - Not significantly impact runtime performance

############################################################
## SCORING AND EVALUATION

We will benchmark on a fixed random seed over several CVRP instances.
Your patch should reduce the average optimal gap in ≥90% of the instances without
increasing total runtime by >3%.

Key considerations for high-quality solutions:
- More efficient route structures (fewer vehicles, shorter routes)
- Better client assignment to routes based on spatial relationships
- Improved handling of capacity constraints
- Preservation of high-quality route segments during crossover
- Better diversity in the generated offspring

############################################################
    
## {self.get_filename()}
```cpp
{get_baseline_code(self.crossover_type) if previous_best_code is None else previous_best_code}
```

## Extra Information:
## DOMAIN KNOWLEDGE: CVRP AND CROSSOVER OPERATIONS

The Selective Route Exchange is a crossover operation for the Capacitated Vehicle Routing Problem (CVRP). The algorithm:
1. Selects routes from two parent solutions
2. Exchanges these routes to create offspring
3. Aims to preserve beneficial route structures while creating new combinations

### Key Optimization Areas to Consider:
- Route selection strategy (which routes to exchange)
- Client-to-route assignment decisions
- Proximity/distance calculations between routes or clients
- Handling of capacity constraints
- Diversity generation in offspring solutions

## Essential Fields and Methods for CVRP Crossover

**ProblemData Key Methods:**
- `numLocations()` - Returns `size_t` total number of locations (depots + clients)
- `numClients()` - Returns `size_t` number of client locations
- `centroid()` - Returns `std::pair<double, double>` center of all client locations
- `client(idx)` - Returns `ProblemData::Client` with coordinates (x, y)

**Route Key Methods:**
- `centroid()` - Returns `std::pair<double, double>` center of route's client locations
- `vehicleType()` - Returns `VehicleType` (size_t) vehicle type index
- `begin()` / `end()` - Iterator support for visiting clients in route
- `size()` - Returns `size_t` number of clients in route
- `visits()` - Returns `std::vector<Client>` all client indices in route order

**Route Construction:**
- `Route(data, visits, vehicleType)` - Constructor taking `std::vector<Client>` visits and vehicle type

**Client Iteration:**
- Routes are iterable containers of `Client` (size_t) indices
- Use range-based for loops: `for (Client c : route)` to access all clients in route
- Client coordinates: `data.client(c).x`, `data.client(c).y`
"""
        if self.module_to_modify == "subpopulation":
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

7. □ You must make at least one modification; DO NOT copy the original code.
   
8. □ Before finalizing, double-check that your modification:
   - Does not introduce new parameters
   - Does not change the function's contract
   - Is focused on improving solution quality, not runtime
   - Is fully compatible with the existing codebase
   - Uses only documented methods and attributes

############################################################
## DELIVERABLES (strict):

A. ≤ 2-sentence summary of the optimization idea, clearly explaining how it improves solution quality.

B. Output the FULL C++ code with your modifications. Mark all changes with "// MODIFY: XXX" comments.

C. Brief explanation of your verification process and why you're confident the modification will:
   - Improve solution quality
   - Maintain compatibility with the existing codebase
   - Not significantly impact runtime performance

############################################################
## SCORING AND EVALUATION

We will benchmark on a fixed random seed over several CVRP instances.
Your patch should reduce the average optimal gap in ≥90% of the instances without
increasing total runtime by >3%.

############################################################

## Extra Information:
## DOMAIN KNOWLEDGE: SUBPOPULATION ALGORITHM AND GENETIC OPERATORS

The SubPopulation class manages a collection of VRP solutions with automatic survivor selection. It maintains solution diversity while preserving high-quality individuals.

### Key Algorithm Components:
1. Population management with automatic purging when exceeding maxPopSize
2. Biased fitness calculation combining cost rank and diversity rank
3. Diversity management using proximity lists and distance measures

### Key Optimization Areas to Consider:
- Fitness calculation strategy (cost vs diversity balance)
- Solution quality assessment beyond simple cost ranking
- Adaptive population management based on convergence state
- Multi-criteria selection considering solution stability
- Enhanced diversity metrics incorporating route-level features

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
{get_baseline_code("subpopulation") if previous_best_code is None else previous_best_code}
```
"""

        return [
            {
                "role": "system", 
                "content": "You are an expert C++ optimization engineer specializing in Vehicle Routing Problems (VRP). Your role is to carefully analyze and improve algorithms while maintaining compatibility with existing code. Focus on incremental improvements that build upon previous optimizations."
            },
            {
                "role": "user",
                "content": question,
            }
        ]
    
    def generate_chat_response(self, messages: List[Dict[str, str]]) -> str:
        response = self.llm.chat(
            messages=messages,
            sampling_params=self.sampling_params
        )
        print(response[0].outputs[0].text.strip())
        return response[0].outputs[0].text.strip()
    
    def initialize_baseline(self, random_seed=42):
        """Initialize baseline results once at the start of evaluation."""
        if self.baseline_results is None:
            print("🔧 First run: initializing baseline results...")
            from evaluate_llm_code import run_baseline_test_with_crossover
            self.baseline_results = run_baseline_test_with_crossover(random_seed, self.iters, self.num_procs, self.crossover_type)
            if self.baseline_results is None:
                raise RuntimeError("❌ Baseline initialization failed; cannot continue evaluation")
            print("✅ Baseline initialized; subsequent evaluations will reuse these results")
    
    def evaluate_model(self, num_samples: int = 5, rollout_rounds: int = 1) -> tuple[List[float], List[str], List[Dict]]:
        """Evaluate the model using chat-format generation with optional multi-round rollout."""
        print(f"Generating {num_samples} responses over {rollout_rounds} rollout round(s)...")

        self.initialize_baseline()
        
        all_rounds_data = []
        best_code_so_far = None
        best_score_so_far = float('inf')
        final_scores = []
        final_responses = []
        
        for round_num in range(rollout_rounds):
            print(f"\n{'='*80}")
            print(f"🔄 Round {round_num + 1}/{rollout_rounds}")
            print(f"{'='*80}")
            
            round_scores = []
            round_responses = []
            round_data = {
                "round": round_num + 1,
                "scores": [],
                "best_score": None,
                "best_code": None,
                "improvement": None
            }
            
            for i in range(num_samples):
                try:
                    print(f"\n{'='*60}")
                    print(f"Round {round_num + 1} - Sample {i+1}/{num_samples}")
                    print(f"{'='*60}")
                    
                    # In round 0 use baseline as context; otherwise feed last round's best code
                    messages = self.create_chat_messages(
                        previous_best_code=best_code_so_far if round_num > 0 else None
                    )
                    generated_code = self.generate_chat_response(messages)
                    code = self._extract_code_block(generated_code)
                    # code = get_baseline_code(self.crossover_type)
                    if code:
                        from .evaluate_llm_code import evaluate_llm_code_with_baseline
                        result = evaluate_llm_code_with_baseline(code, self.baseline_results, problem_types=self.problem_types, iters=self.iters, num_procs=self.num_procs, crossover_type=self.crossover_type, module_to_modify=self.module_to_modify)
                        
                        # Handle both detailed-dict and legacy scalar return formats
                        if isinstance(result, dict):
                            score = result.get("average_gap", result.get("average_cost", -1))
                            detailed_results = result
                        else:
                            score = result
                            detailed_results = {"average_gap": score, "datasets": {}, "instance_details": {}}
                        
                        round_scores.append(score)
                        round_responses.append({"code": code, "detailed_results": detailed_results})
                        print(f"✅ Score (Gap): {score:.6f}")

                        if "gaps" in detailed_results:
                            print("📊 Gap breakdown:")
                            llm_avg = detailed_results.get("llm_average_cost", 0)
                            baseline_avg = detailed_results.get("baseline_average_cost", 0)
                            print(f"   LLM avg cost:      {llm_avg:.4f}")
                            print(f"   Baseline avg cost: {baseline_avg:.4f}")
                            print(f"   Average gap:       {score:.6f}")

                            gaps = detailed_results["gaps"]
                            positive_gaps = [g for g in gaps.values() if g > 0]
                            negative_gaps = [g for g in gaps.values() if g < 0]
                            print(f"   Improved datasets: {len(negative_gaps)}/{len(gaps)} (gap<0)")
                            print(f"   Degraded datasets: {len(positive_gaps)}/{len(gaps)} (gap>0)")

                        if "instance_details" in detailed_results:
                            print("📋 Instance details:")
                            for vrp_type, sizes in detailed_results["instance_details"].items():
                                for size, info in sizes.items():
                                    print(f"   {vrp_type}/{size}: avg cost={info['average_cost']:.4f}, instances={info['num_instances']}")

                        # Lower gap is better; negative gap means improvement over baseline
                        if score < best_score_so_far:
                            best_score_so_far = score
                            best_code_so_far = code
                            print(f"🆕 New best score: {score:.4f}")
                    else:
                        print(f"❌ No valid code block found")
                        round_scores.append(-1)
                        round_responses.append({"code": generated_code, "detailed_results": {}})
                        
                except Exception as e:
                    print(f"❌ Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    round_scores.append(-1)
                    round_responses.append({"code": "", "detailed_results": {}})
            
            # Record round data
            valid_scores = [s for s in round_scores if s >= 0]
            if valid_scores:
                round_best_score = min(valid_scores)
                round_best_idx = round_scores.index(round_best_score)
                round_best_response = round_responses[round_best_idx]
                round_best_code = round_best_response["code"] if isinstance(round_best_response, dict) else round_best_response
                
                round_data.update({
                    "scores": round_scores,
                    "best_score": round_best_score,
                    "best_code": round_best_code,
                })
                
                print(f"\n📊 Round {round_num + 1} summary:")
                print(f"   Best score: {round_best_score:.4f}")
            else:
                print(f"\n⚠️ Round {round_num + 1}: no valid results")
            
            all_rounds_data.append(round_data)
            
            # Save final data on the last round
            if round_num == rollout_rounds - 1:
                final_scores = round_scores
                final_responses = round_responses
        
        return final_scores, final_responses, all_rounds_data
    
    def _extract_code_block(self, response: str) -> str:
        """Extract the first C++ code block from the response."""
        code = response.split("```cpp")[-1].split("```")[0]
        return code.strip() if code.strip() else ""


def find_all_models(directory):
    """Find all subdirectories that contain at least one safetensors file."""
    models = []
    for root, dirs, files in os.walk(directory):
        if any(file.endswith('.safetensors') for file in files):
            rel_path = os.path.relpath(root, directory)
            if rel_path != '.':
                models.append(root)
    return models

def main():
    parser = argparse.ArgumentParser(description="Evaluate model(s) with vLLM")
    parser.add_argument("model_path", help="path to model checkpoint or parent directory")
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="number of samples per model")
    parser.add_argument("--output_file", help="output file path (JSON)")
    parser.add_argument("--problem_types", type=str, default="['CVRP_all']", help="problem types list")
    parser.add_argument("--iters", type=int, default=800, help="max HGS iterations")
    parser.add_argument("--num_procs", type=int, default=16, help="parallel worker processes")
    parser.add_argument("--temperature", type=float, default=1, help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=1, help="top-p for sampling")
    parser.add_argument("--top_k", type=int, default=1, help="top-k for sampling")
    parser.add_argument("--multi_model", action="store_true", help="scan directory for all model checkpoints")
    parser.add_argument("--rollout_rounds", type=int, default=1, help="number of refinement rounds (>1 enables multi-round rollout)")
    parser.add_argument("--crossover_type", type=str, default="mtsp", help="crossover type: tsp or mtsp")
    parser.add_argument("--module_to_modify", type=str, default="subpopulation", help="module to optimize: subpopulation or crossover")
    args = parser.parse_args()
    print(f"Args: {args}")
    
    SAMPLING_CONFIG.update({
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
    })
    
    model_paths = []
    if args.multi_model or os.path.isdir(args.model_path):
        model_paths = find_all_models(args.model_path)
        if not model_paths:
            model_paths = [args.model_path]
            print(f"No sub-models found; evaluating single model: {args.model_path}")
        else:
            print(f"Found {len(model_paths)} model(s):")
            for i, path in enumerate(model_paths, 1):
                print(f"  {i}. {path}")
    else:
        model_paths = [args.model_path]

    all_results = {}
    best_model = None
    best_score = float('-inf')
    
    try:
        for model_path in model_paths:
            print(f"\n{'='*80}")
            print(f"Evaluating model: {model_path}")
            print(f"{'='*80}")
            
            try:
                evaluator = VLLMEvaluator(model_path, args.problem_types, args.iters, args.num_procs, args.crossover_type, args.module_to_modify)
                scores, responses, rollout_data = evaluator.evaluate_model(
                    args.num_samples, 
                    rollout_rounds=args.rollout_rounds
                )
                
                if scores:
                    max_score = max(scores)
                    mean_score = np.mean(scores)
                    
                    all_results[model_path] = {
                        "scores": scores,
                        "max_score": float(max_score),
                        "mean_score": float(mean_score),
                        "num_samples": len(scores),
                        "rollout_data": rollout_data,
                        "rollout_rounds": args.rollout_rounds
                    }
                    
                    print(f"✅ {os.path.basename(model_path)}: evaluation complete")
                    print(f"   Max score:  {max_score:.4f}")
                    print(f"   Mean score: {mean_score:.4f}")
                    if max_score > best_score:
                        best_score = max_score
                        best_model = model_path
                        
                else:
                    print(f"❌ {model_path}: no valid evaluation results")

            except Exception as e:
                print(f"❌ Error evaluating {model_path}: {e}")
                import traceback
                traceback.print_exc()
                all_results[model_path] = {
                    "error": str(e),
                    "scores": [],
                    "max_score": float('-inf'),
                    "mean_score": float('-inf'),
                    "num_samples": 0
                }
        
        print(f"\n{'='*80}")
        print("🎉 All evaluations complete!")
        print(f"{'='*80}")

        if best_model:
            print(f"🏆 Best model: {os.path.basename(best_model)}")
            print(f"📊 Best score: {best_score:.4f}")
            print(f"📁 Path:       {best_model}")
        else:
            print("⚠️ No valid evaluation results found")

        print(f"\n📈 Results summary:")
        print("-" * 60)
        for model_path, result in all_results.items():
            model_name = os.path.basename(model_path)
            if "error" in result:
                print(f"{model_name:20} | error: {result['error']}")
            else:
                print(f"{model_name:20} | max: {result['max_score']:.4f} | mean: {result['mean_score']:.4f}")

        if args.output_file:
            final_results = {
                "total_models": len(model_paths),
                "best_model": best_model,
                "best_score": float(best_score) if best_score != float('-inf') else None,
                "best_model_name": os.path.basename(best_model) if best_model else None,
                "rollout_rounds": args.rollout_rounds,
                "all_results": all_results,
                "evaluation_summary": {
                    "models_evaluated": len([r for r in all_results.values() if "error" not in r and r.get("scores")]),
                    "models_with_errors": len([r for r in all_results.values() if "error" in r]),
                    "total_models": len(model_paths),
                    "rollout_strategy": "multi_round" if args.rollout_rounds > 1 else "single_round"
                }
            }
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            print(f"\n📁 Results saved to: {args.output_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()