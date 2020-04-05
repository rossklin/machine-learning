#include "game.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "agent.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

game::game(player_table pl) : players(pl) {
  static int idc = 0;
  MutexType m;

  m.Lock();
  game_id = idc++;
  m.Unlock();

  max_turns = 100;
  enable_output = false;
}

void game::initialize() {
  turns_played = 0;
}

hm<int, vector<record>> game::play(int epoch) {
  hm<int, vector<record>> res;

  for (turns_played = 0; turns_played < max_turns && !finished(); turns_played++) {
    record_table rect = increment();
    for (auto x : rect) res[x.first].push_back(x.second);
  }

  // add reward for winning team
  if (winner > -1) {
    float reward = winner_reward(epoch);
    for (auto x : players) {
      if (x.second->team == winner) res[x.first].back().reward += reward;
    }
  }

  return res;
}

void game::reset() {
  turns_played = 0;
  winner = -1;
}

vector<int> game::team_clone_ids(int tid) const {
  vector<int> res;
  for (auto x : players) {
    if (x.second->team == tid) res.push_back(x.first);
  }
  return res;
}

vec game::vectorize_input(choice_ptr c, int pid) const {
  vec s = vectorize_state(pid);
  vec x = vectorize_choice(c, pid);
  x.insert(x.end(), s.begin(), s.end());
  return x;
}