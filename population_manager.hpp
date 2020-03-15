#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "types.hpp"

class population_manager {
 public:
  int popsize;
  float preplim;
  float simple_score_limit;
  agent_f gen;
  std::vector<agent_ptr> pop;

  population_manager(int popsize, agent_f gen, float plim);
  std::string pop_stats(std::string row_prefix) const;
  std::vector<agent_ptr> topn(int n) const;
  void check_gg(game_generator_ptr gg) const;

  virtual void prepare_epoch(int epoch, game_generator_ptr ggen);
  virtual void evolve(game_generator_ptr ggen);
};

typedef population_manager default_population_manager;