#pragma once

#include "game_generator.hpp"

class pod_game_generator : public game_generator {
  game_ptr generate_starting_state(std::vector<agent_ptr> p);
  double winner_reward(int epoch);
  agent_ptr generate_player();
  agent_ptr generate_refbot();
};