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

// Find the latest time_point in the ordered vector fired_at that is no later than t
time_point Node::last_fired_before(time_point t) const
{
    if (fired_at.empty())
    {
        return -1;
    }

    int top = fired_at.size() - 1; // Highest possible index
    int bottom = 0;                // Lowest possible index

    while (top > bottom)
    {
        int idx = bottom + (top - bottom + 1) / 2; // idx guaranteed greater than bottom
        if (fired_at[idx] <= t)
        {
            bottom = idx;
        }
        else
        {
            top = idx - 1;
        }
    }

    time_point t2 = fired_at[bottom];

    if (t2 <= t)
    {
        return t2;
    }
    else
    {
        return -1;
    }
}

bool Node::did_fire_at(time_point t) const
{
    return last_fired_before(t) == t;
};
