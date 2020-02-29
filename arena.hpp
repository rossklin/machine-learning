#pragma once

#include <functional>
#include <vector>

#include "types.hpp"

// ARENA
template <typename G, typename T, typename P>
class arena {
  G ggn;
  T trm;
  P pop;

 public:
  void evolution(int threads, int ngames, int ppt, int tpg);
  void write_stats(int epoch) const;
};