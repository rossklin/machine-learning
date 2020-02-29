#pragma once

#include <functional>
#include <memory>
#include <vector>

template <typename A>
class population_manager {
 public:
  typedef A::ptr agent_ptr;
  int popsize;
  std::vector<agent_ptr> pop;

  population_manager(int popsize);
  agent_ptr prepared_player(std::function<agent_ptr()> gen, float plim) const;
  std::vector<agent_ptr> prepare_n(std::function<agent_ptr()> gen, int n, float plim) const;
  std::string pop_stats(std::string row_prefix) const;
  std::vector<agent_ptr> topn(int n) const;
  virtual void prepare_epoch(int epoch);
  virtual void evolve();
};