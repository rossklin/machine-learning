#pragma once
#include <string>
#include <vector>

#include "types.hpp"

typedef hm<int, agent_ptr> player_table;
// typedef std::function<std::shared_ptr<agent>()> player_sampler;
// typedef std::function<std::shared_ptr<game>(std::vector<std::shared_ptr<agent> >)> game_sampler;

class game {
 public:
  bool enable_output;
  int winner;
  int nr_of_teams;
  int ppt;
  player_table players;
  std::vector<record_table> results;

  game(int teams, int ppt);

  virtual game_ptr generate_starting_state(std::vector<agent_ptr> p) = 0;
  virtual record_table increment() = 0;
  virtual bool finished() = 0;
  virtual std::string end_stats(int pid, int pid2) = 0;
  virtual int select_winner() = 0;
  virtual double score_simple(int pid) = 0;
  virtual double winner_reward(int epoch) = 0;
  virtual void reset() = 0;
  virtual agent_ptr generate_player() = 0;
  virtual agent_ptr generate_refbot() = 0;
  virtual std::vector<choice_ptr> generate_choices(agent_ptr a) = 0;
  virtual vec vectorize_choice(choice_ptr c, int pid) = 0;
  virtual float reward_win() = 0;

  void play();
  void train(int pid, int filter_pid = -1, int filter_tid = -1);
  game_ptr team_bots_vs(agent_ptr a);
  std::vector<agent_ptr> make_teams(std::vector<agent_ptr> ps);
  std::function<vec()> generate_input_sampler();
  int choice_dim();
};