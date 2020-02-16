#pragma once

#include "types.hpp"

class tournament {
 public:
  virtual void run(population_manager_ptr pm) = 0;
};