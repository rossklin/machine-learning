#pragma once

#include <functional>
#include <vector>

#include "types.hpp"

// ARENA
template <typename G, typename T, typename P>
class arena {
 public:
  G ggn;
  T trm;
  P pop;
  typedef G::agent_ptr agent_ptr;
  double mate_score(agent_ptr parent1, agent_ptr x) const;

  arena(int teams, int ppt);
  void evolution(int threads, int ngames);
  void write_stats(int epoch) const;
};