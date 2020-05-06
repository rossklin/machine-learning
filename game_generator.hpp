#pragma once

#include <set>

#include "team.hpp"
#include "types.hpp"

class game_generator {
 public:
  agent_f refbot_generator;
  int nr_of_teams;
  int ppt;
  int prep_npar;
  int max_turns;

  game_generator(int teams, int ppt, agent_f refbot_generator);

  virtual game_ptr generate_starting_state(std::vector<team> p) const = 0;
  virtual std::set<int> required_inputs() const = 0;

  agent_ptr prepared_player(input_sampler isam, agent_f gen, float plim) const;
  std::vector<team> prepare_n(agent_f gen, int n, float plim) const;
  game_ptr team_bots_vs(team a) const;
  // std::vector<agent_ptr> make_teams(std::vector<agent_ptr> ps) const;
  std::function<vec()> generate_input_sampler() const;
  int choice_dim() const;
};