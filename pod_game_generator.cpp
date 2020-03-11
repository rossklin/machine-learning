#include "pod_game_generator.hpp"
#include "pod_game.hpp"

using namespace std;

game_ptr pod_game_generator::generate_starting_state(std::vector<agent_ptr> p) const {
  player_table pl;
  for (auto a : p) pl[a->id] = a;
  game_ptr g(new pod_game(pl));
  g->initialize();
  return g;
}
