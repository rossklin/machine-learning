#pragma once
#include <vector>
#include <set>

#include "util.hpp"

struct Node
{
    float energy;
    float inhibition;
    float energy_uptake;
    float firepower;
    float modification_tracker;
    float sadness;
    std::vector<time_point> fired_at;
    std::vector<node_index> parents;

    Node();

    std::set<time_point> fired_at_set() const;
};