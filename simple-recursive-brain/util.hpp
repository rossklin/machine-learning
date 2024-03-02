#pragma once

#include <random>

std::default_random_engine &get_engine();
int random_int(int a, int b);
float random_float(float a, float b);

typedef int time_point;
typedef int node_index;
