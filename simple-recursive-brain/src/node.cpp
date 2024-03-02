#include "node.hpp"

Node::Node()
{
    energy = 0;
    inhibition = 0;
    energy_uptake = 0;
    firepower = 0;
    modification_tracker = 0;
}

std::set<int> Node::fired_at_set() const
{
    return std::set<int>(fired_at.begin(), fired_at.end());
}