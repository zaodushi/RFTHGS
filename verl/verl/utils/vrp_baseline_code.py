import os

module_to_modify = os.getenv("module_to_modify", "crossover")

if module_to_modify == "subpopulation" or module_to_modify == "subpopulation_new_prompt":
    BASELINE_CODE = """
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

    for (auto &other : items_)  // update distance to other solutions
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

    items_.push_back(item);  // add solution

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
        if (duplicate == items_.end())  // there are no more duplicates
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

    std::vector<std::pair<double, size_t>> diversity;
    for (size_t costRank = 0; costRank != size(); costRank++)
    {
        auto const dist = items_[byCost[costRank]].avgDistanceClosest();
        diversity.emplace_back(-dist, costRank);  // higher is better
    }

    std::stable_sort(diversity.begin(), diversity.end());

    auto const popSize = static_cast<double>(size());
    auto const numElite = std::min(params.numElite, size());
    auto const divWeight = 1 - numElite / popSize;

    for (size_t divRank = 0; divRank != size(); divRank++)
    {
        auto const costRank = diversity[divRank].second;
        auto const idx = byCost[costRank];
        items_[idx].fitness = (costRank + divWeight * divRank) / (2 * popSize);
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
elif os.environ.get("use_pyvrp_code") == "True":
    BASELINE_CODE = """
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
else:
    BASELINE_CODE = """
#include "selective_route_exchange.h"

#include "DynamicBitset.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

using Client  = size_t;
using Clients = std::vector<Client>;
using Route   = pyvrp::Route;
using Routes  = std::vector<Route>;

namespace pyvrp::crossover
{
/*
 * Minimal selective-route-exchange:
 * 1. Take numMovedRoutes consecutive routes from parent B (starting index given by startIndices.second).
 * 2. Remove the clients of these routes from parent A's routes, then append the entire routes to child 1.
 * 3. Do the symmetric operation to obtain child 2.
 * 4. Evaluate penalised costs and return the better one.
 */
pyvrp::Solution selectiveRouteExchange(
    std::pair<Solution const *, Solution const *> const &parents,
    ProblemData const &data,
    CostEvaluator const &costEvaluator,
    std::pair<size_t, size_t> const &startIndices,
    size_t const numMovedRoutes)
{
    auto const *parentA = parents.first;
    auto const *parentB = parents.second;

    size_t const nRoutesA = parentA->numRoutes();
    size_t const nRoutesB = parentB->numRoutes();

    if (numMovedRoutes == 0 || numMovedRoutes > std::min(nRoutesA, nRoutesB))
        throw std::invalid_argument("numMovedRoutes out of range.");

    size_t const startA = startIndices.first  % nRoutesA;
    size_t const startB = startIndices.second % nRoutesB;

    /* ------ helper lambda that builds a single child ------ */
    auto makeChild = [&](Routes const &donorRoutes,
                         Routes const &receiverRoutes,
                         size_t donorStartIdx) -> Solution
    {
        /* 1) Take numMovedRoutes consecutive routes from the donor */
        Routes moved;
        moved.reserve(numMovedRoutes);
        for (size_t r = 0; r < numMovedRoutes; ++r)
            moved.push_back(donorRoutes[(donorStartIdx + r) % donorRoutes.size()]);

        /* 2) Mark all clients of these routes in a DynamicBitset */
        DynamicBitset mark(data.numLocations());
        for (Route const &route : moved)
            for (Client c : route)
                mark[c] = true;             // set bit

        /* 3) Copy receiver's routes and delete marked clients */
        Routes childRoutes;
        childRoutes.reserve(receiverRoutes.size() + moved.size());

        for (Route const &route : receiverRoutes)
        {
            Clients kept;
            kept.reserve(route.size());

            for (Client c : route)
                if (!mark[c])
                    kept.push_back(c);

            if (!kept.empty())
                childRoutes.emplace_back(data, kept, route.vehicleType());
        }

        /* 4) Append the donor routes to the end */
        for (Route const &route : moved)
            childRoutes.emplace_back(data,
                                     Clients(route.begin(), route.end()),
                                     route.vehicleType());

        return Solution(data, childRoutes);
    };

    /* ------ build two children and return the best ------ */
    Solution child1 = makeChild(parentB->routes(), parentA->routes(), startB);
    Solution child2 = makeChild(parentA->routes(), parentB->routes(), startA);

    return costEvaluator.penalisedCost(child1) <
           costEvaluator.penalisedCost(child2)
           ? child1
           : child2;
}
}  // namespace pyvrp::crossover
"""