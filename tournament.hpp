#pragma once

#include "types.hpp"

class tournament {
 public:
  virtual void run(population_manager_ptr pm, int epoch) = 0;
};