#pragma once

#include <memory>

#include "agent.hpp"
#include "choice.hpp"
#include "game.hpp"
#include "pod_agent.hpp"
#include "types.hpp"

class pod_choice : public choice {
 public:
  double angle;
  double thrust;
  bool boost;
  bool shield;
  bool validate();
  int vector_dim();
};

namespace pod_game_parameters {
constexpr double width = 16000;
constexpr double height = 9000;
constexpr double checkpoint_radius = 600;
constexpr double pod_radius = 400;
constexpr double angular_speed = 0.314;
constexpr double friction = 0.85;
constexpr double pod_mass = 1;
};  // namespace pod_game_parameters

class pod_game : public game, public std::enable_shared_from_this<pod_game> {
 protected:
  hm<int, double> htable();
  hm<int, double> ttable();
  double pod_distance_travelled(int pid);

  bool did_finish;
  int run_laps;
  std::vector<point> checkpoint;

  point get_checkpoint(int idx) const;
  hm<int, pod_agent::ptr> typed_agents;

 public:
  pod_game(player_table pl);
  void initialize();
  record_table increment();
  bool finished();
  std::string end_stats();
  int select_winner();
  double score_simple(int pid);
  void reset();
  std::vector<choice_ptr> generate_choices(agent_ptr a);
  vec vectorize_choice(choice_ptr c, int pid) const override;
  vec vectorize_state(int pid) const override;
  double winner_reward(int epoch);
};
