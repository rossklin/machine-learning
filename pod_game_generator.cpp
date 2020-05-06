#include "pod_game_generator.hpp"

#include <cassert>

#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;

pod_game_generator::pod_game_generator(int teams, int ppt, agent_f refbot_generator) : game_generator(teams, ppt, refbot_generator) {
  max_turns = 300;
}

game_ptr pod_game_generator::generate_starting_state(std::vector<agent_ptr> p) const {
  player_table pl;
  for (auto a : p) pl[a->id] = a;
  game_ptr g(new pod_game(pl));
  g->initialize();

  // check that g has right number of teams and players
  hm<int, int> tc;
  for (auto x : g->players) tc[x.second->team]++;

  assert(tc.size() == nr_of_teams);
  for (auto x : tc) assert(x.second == ppt);

  return g;
}

// require the first 5 inputs be included for a functioning evaluator
set<int> pod_game_generator::required_inputs() const {
  assert(nr_of_teams == 2);
  set<int> buf{0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19};
  if (ppt > 1) buf = set_union(buf, set<int>({31, 32, 44, 45}));
  return buf;
}
