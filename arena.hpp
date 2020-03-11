#pragma once

#include <functional>
#include <vector>

#include "types.hpp"

// ARENA
template <typename T, typename P>
class arena {
 public:
  game_generator_ptr ggn;
  T trm;
  P pop;

  arena(game_generator_ptr g);
  double mate_score(agent_ptr parent1, agent_ptr x) const;
  void evolution(int threads, int ngames);
  void write_stats(int epoch) const;
};