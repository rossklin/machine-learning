#pragma once
#include <string>
#include <vector>

#include "types.hpp"

typedef hm<int, agent_ptr> player_table;

class game {
 public:
  int game_id;
  bool enable_output;
  int winner;
  player_table players;

  game(player_table pl);
  virtual void initialize();

  virtual double winner_reward(int epoch) = 0;
  virtual record_table increment() = 0;
  virtual bool finished() = 0;
  virtual std::string end_stats(int pid, int pid2) = 0;
  virtual int select_winner() = 0;
  virtual double score_simple(int pid) = 0;
  virtual void reset() = 0;
  virtual std::vector<choice_ptr> generate_choices(agent_ptr a) = 0;
  virtual vec vectorize_choice(choice_ptr c, int pid) = 0;

  hm<int, vector<record>> play(int epoch);
};