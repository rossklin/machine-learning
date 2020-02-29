#include <cassert>
#include <functional>
#include <vector>

#include "agent.hpp"
#include "game.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

template <typename A>
game<A>::game(player_table pl) : players(pl) {}

template <typename A>
void game<A>::initialize() {}  // todo: what was this for?

template <typename A>
hm<int, vector<record>> game<A>::play(int epoch) {
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

template <typename A>
choice_ptr game<A>::select_choice(agent_ptr a) {
  auto opts = generate_choices(shared_from_this());
  for (auto opt : opts) opt->value_buf = a->evaluate_choice(vectorize_choice(opt, a->id));
  return csel.select(opts);
}