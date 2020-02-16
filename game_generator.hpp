#pragma once

#include "types.hpp"

class game_generator {
 public:
  int nr_of_teams;
  int ppt;

  game_generator(int teams, int ppt);
  virtual game_ptr generate_starting_state(std::vector<agent_ptr> p) = 0;
  virtual double winner_reward(int epoch) = 0;
  virtual agent_ptr generate_player() = 0;
  virtual agent_ptr generate_refbot() = 0;

  game_ptr team_bots_vs(agent_ptr a);
  std::vector<agent_ptr> make_teams(std::vector<agent_ptr> ps);
  std::function<vec()> generate_input_sampler();
  int choice_dim();
};