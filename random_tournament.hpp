#pragma once

#include "tournament.hpp"

// tournament where each player plays one random game
class random_tournament : public tournament {
  int game_rounds;

 public:
  random_tournament(int gr = 100);
  void run(population_manager_ptr pm, game_generator_ptr gg, int epoch);
};