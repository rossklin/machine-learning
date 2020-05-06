#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "types.hpp"

struct team {
  double score;
  int last_rank;
  int rank;
  bool was_protected;
  int id;

  std::vector<agent_ptr> players;

  team(std::vector<agent_ptr> ps);
  void deserialize(std::stringstream &ss);
  std::string serialize() const;
  double simple_score() const;
  void set_exploration_rate(double q);
};
