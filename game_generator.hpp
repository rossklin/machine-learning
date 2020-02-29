#pragma once

#include "types.hpp"

template <typename GAME_CLASS, typename REFBOT_CLASS>
class game_generator {
 public:
  int nr_of_teams;
  int ppt;

  game_generator(int teams, int ppt);
  game_ptr generate_starting_state(std::vector<agent_ptr> p);
  game_ptr team_bots_vs(agent_ptr a);
  std::vector<agent_ptr> make_teams(std::vector<agent_ptr> ps);
  std::function<vec()> generate_input_sampler();
  int choice_dim();
};