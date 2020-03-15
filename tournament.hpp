#pragma once

#include "types.hpp"

class tournament {
 public:
  virtual void run(population_manager_ptr pm, game_generator_ptr gg, int epoch) = 0;
};