#include <cassert>
#include <functional>
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

  for (turns_played = 0; turns_played < max_turns && !finished(); turns_played++) {
    record_table rect = increment();
    for (auto x : rect) res[x.first].push_back(x.second);
  }

  // add reward for winning
  if (winner > -1) {
    res[winner].back().reward += winner_reward(epoch);
  }

  return res;
}

void game::reset() {
  turns_played = 0;
  winner = -1;
}