#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "agent.hpp"
#include "game.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

game::game(player_table pl) : players(pl) {
  max_turns = 100;
}

void game::initialize() {
  turns_played = 0;
}  // todo: what was this for?

hm<int, vector<record>> game::play(int epoch) {
  hm<int, vector<record>> res;

  cout << "game::play: players: ";
  for (auto x : players) {
    cout << x.first << " (" << x.second->id << "), ";
  }
  cout << endl;

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

vector<int> game::team_pids(int tid) const {
  vector<int> res;
  for (auto x : players) {
    if (x.second->team == tid) res.push_back(x.first);
  }
  return res;
}