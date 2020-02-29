#include <cassert>
#include <functional>
#include <vector>

#include "agent.hpp"
#include "game.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

game::game(player_table pl) : players(pl) {}

void game::initialize() {}  // todo: what was this for?

hm<int, vector<record>> game::play(int epoch) {
  hm<int, vector<record>> res;

  while (!finished()) {
    record_table rect = increment();
    for (auto x : rect) res[x.first].push_back(x.second);
  }

  // add reward for winning
  if (winner > -1) {
    res[winner].back().reward += winner_reward(epoch);
  }

  return res;
}
