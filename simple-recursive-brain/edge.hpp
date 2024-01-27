#pragma once

struct Edge
{
    float width;
    bool is_inhibitor;

    int get_type_sign() const;
};