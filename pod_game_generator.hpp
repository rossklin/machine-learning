#pragma once

#include "game_generator.hpp"
#include "types.hpp"

class pod_game_generator : public game_generator {
 public:
  pod_game_generator(int teams, int ppt, agent_f refbot_generator);
  game_ptr generate_starting_state(std::vector<agent_ptr> p) const override;
};