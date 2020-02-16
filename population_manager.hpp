#pragma once

#include <vector>
#include "types.hpp"

class population_manager {
 public:
  int popsize;
  std::vector<agent_ptr> pop;

  population_manager(int popsize);
  std::string pop_stats(std::string row_prefix) const;
  std::vector<agent_ptr> topn(int n) const;
  virtual void prepare_epoch(int epoch);
  virtual void evolve();
};