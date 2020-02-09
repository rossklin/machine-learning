#pragma once

#include <functional>
#include <vector>

#include "types.hpp"

// ARENA
class arena {
  game_ptr base_game;
  int ppt, tpg, ppg;
  int choice_dim;
  int threads;
  float simple_score_limit;
  float preplim;

  std::vector<record_table> play_game(game_ptr start);
  agent_ptr prepared_player(std::function<agent_ptr()> generator, float plim = 1);
  std::vector<agent_ptr> prepare_n(std::function<agent_ptr()> generator, int n, float plim);

 public:
  arena(game_ptr bg, int threads, int ppt, int tpg, float preplim);
  void evolution(int ngames = 64);
};