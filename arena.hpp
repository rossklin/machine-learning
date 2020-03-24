#pragma once

#include <string>

#include "types.hpp"

void evolution(game_generator_ptr gg, tournament_ptr t, population_manager_ptr p, int threads, std::string loadfile = "");
