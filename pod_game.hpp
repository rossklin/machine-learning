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

class pod_game : public game, public std::enable_shared_from_this<pod_game> {
 protected:
  hm<int, double> htable();
  hm<int, double> ttable();
  point get_checkpoint(int idx);

  static constexpr double width = 16000;
  static constexpr double height = 9000;
  static constexpr double checkpoint_radius = 600;
  static constexpr double pod_radius = 400;
  static constexpr double angular_speed = 0.314;
  static constexpr double friction = 0.85;
  static constexpr double pod_mass = 1;
  int game_id;
  bool did_finish;
  int run_laps;
  int tree_depth;
  std::vector<point> checkpoint;

  hm<int, pod_agent_ptr> get_typed_agents();

 public:
  pod_game(int teams, int ppt, int tree_depth);

  // static agent_ptr simple_pod_agent();
  // static agent_ptr rbf_pod_agent();
  // static agent_ptr tree_pod_agent();
  game_ptr generate_starting_state(std::vector<agent_ptr> p);

  record_table increment();
  bool finished();
  std::string end_stats(int pid, int pid2);
  int select_winner();
  double winner_reward(int epoch);
  double score_simple(int pid);
  void reset();
  agent_ptr generate_player();
  agent_ptr generate_refbot();
  std::vector<choice_ptr> generate_choices(agent_ptr a);
  vec vectorize_choice(choice_ptr c, int pid);
  float reward_win();
};

typedef std::shared_ptr<pod_game> pod_game_ptr;
typedef std::shared_ptr<pod_choice> pod_choice_ptr;
