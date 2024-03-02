#include "node.hpp"

Node::Node()
{
    energy = 0;
    inhibition = 0;
    energy_uptake = 0;
    firepower = 0;
    modification_tracker = 0;
    sadness = 0;
}

std::set<time_point> Node::fired_at_set() const
{
    return std::set<time_point>(fired_at.begin(), fired_at.end());
}