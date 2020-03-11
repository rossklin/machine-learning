#pragma once
#include <string>
#include <vector>

#include "types.hpp"

class game {
 public:
  int game_id;
  bool enable_output;
  int winner;
  int turns_played;
  int max_turns;
  player_table players;

  game(player_table pl);
  virtual void initialize();
  virtual void reset();

  virtual double winner_reward(int epoch) = 0;
  virtual record_table increment() = 0;
  virtual bool finished() = 0;
  virtual std::string end_stats(int pid, int pid2) = 0;
  virtual int select_winner() = 0;
  virtual double score_simple(int pid) = 0;
  virtual std::vector<choice_ptr> generate_choices(agent_ptr a) = 0;
  virtual vec vectorize_choice(choice_ptr c, int pid) = 0;

  hm<int, std::vector<record>> play(int epoch);
  choice_ptr select_choice(agent_ptr a);
};