#pragma once

#include <functional>
#include <vector>

#include "types.hpp"

void evolution(game_generator_ptr gg, tournament_ptr t, population_manager_ptr p, int threads = 6);
