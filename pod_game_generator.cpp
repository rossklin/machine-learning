#include <cassert>

#include "pod_game.hpp"
#include "pod_game_generator.hpp"

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
