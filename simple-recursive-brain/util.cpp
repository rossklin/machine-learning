#include "util.hpp"

using namespace std;

default_random_engine &get_engine()
{
    static random_device rd; // a seed source for the random number engine
    static default_random_engine gen(rd());
    return gen;
}

int random_int(int a, int b)
{
    uniform_int_distribution<> distrib(a, b);
    return distrib(get_engine());
}

float random_float(float a, float b)
{
    uniform_real_distribution<> distrib(a, b);
    return distrib(get_engine());
}