#pragma once

#include <functional>
#include <vector>

#include "types.hpp"

// ARENA
class arena {
  game_generator_ptr ggn;
  tournament_ptr trm;
  population_manager_ptr pop;

 public:
  arena(game_generator_ptr g, tournament_ptr t, population_manager_ptr p);
  void evolution(int threads, int ngames, int ppt, int tpg);
  void write_stats(int epoch) const;
};