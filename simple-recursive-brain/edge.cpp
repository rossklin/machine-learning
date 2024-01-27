#include "edge.hpp"

int Edge::get_type_sign() const
{
    return is_inhibitor ? -1 : 1;
}