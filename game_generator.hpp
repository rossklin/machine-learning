#pragma once

#include "types.hpp"

class game_generator {
 public:
  agent_f player_generator;
  agent_f refbot_generator;
  int nr_of_teams;
  int ppt;

  game_generator(int teams, int ppt, agent_f player_generator, agent_f refbot_generator);

  virtual game_ptr generate_starting_state(std::vector<agent_ptr> p) const = 0;

  agent_ptr prepared_player(agent_f gen, float plim) const;
  std::vector<agent_ptr> prepare_n(agent_f gen, int n, float plim) const;
  game_ptr team_bots_vs(agent_ptr a) const;
  std::vector<agent_ptr> make_teams(std::vector<agent_ptr> ps) const;
  std::function<vec()> generate_input_sampler() const;
  int choice_dim() const;
};