#pragma once

#include <vector>
#include "types.hpp"

class population_manager {
 public:
  int popsize;
  std::vector<agent_ptr> pop;

  population_manager(int popsize);
  virtual void prepare_epoch();
  virtual void evolve();
  std::string pop_stats() const;
};