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

  point get_checkpoint(int idx) const;

 public:
  std::vector<point> checkpoint;
  std::vector<pod_agent::ptr> typed_agents;

  pod_game(hm<int, team> ts);
  void initialize() override;
  void setup_from_input(std::istream &s) override;
  record_table increment(std::string row_prefix = "") override;
  bool finished() override;
  std::string end_stats() override;
  int select_winner() override;
  double score_simple(int pid) override;
  void reset() override;
  std::vector<choice_ptr> generate_choices(agent_ptr a) override;
  vec vectorize_choice(choice_ptr c, int pid) const override;
  vec vectorize_state(int pid) const override;
  double winner_reward(int epoch) override;
  int agent_team(int pid) const;
};
