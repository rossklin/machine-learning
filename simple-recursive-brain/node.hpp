#pragma once
#include <vector>
#include <set>

struct Node
{
    float energy;
    float inhibition;
    float energy_uptake;
    float firepower;
    float modification_tracker;
    std::vector<int> fired_at;
    std::vector<int> parents;

    Node();

    std::set<int> fired_at_set() const;
};