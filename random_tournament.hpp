#pragma once

#include "tournament.hpp"

// tournament where each player plays one random game
class random_tournament : public tournament {
 public:
  void run(population_manager_ptr pm);
};