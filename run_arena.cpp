#include <cstring>

#include "arena.hpp"
#include "evaluator.hpp"
#include "game_generator.hpp"
#include "pod_game.hpp"
#include "population_manager.hpp"
#include "random_tournament.hpp"
#include "simple_pod_evaluator.hpp"

using namespace std;

int main(int argc, char **argv) {
  int threads = 6;
  int ngames = 64;
  int ppt = 2;
  int tpg = 2;
  int tree_depth = 10;
  float preplim = 0.5;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "debug")) {
      threads = 1;
      tree_depth = 5;
      ngames = 4;
      preplim = 0.01;
    } else if (!strcmp(argv[i], "quick")) {
      threads = 5;
      tree_depth = 7;
      ngames = 10;
      preplim = 0.2;
    } else if (!strcmp(argv[i], "threads")) {
      threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "ngames")) {
      ngames = atoi(argv[++i]);
    }
  }

  typedef pod_agent<simple_pod_evaluator> refbot_t;
  typedef pod_agent<tree_evaluator> agent_t;
  typedef game_generator<pod_game<agent_t>, refbot_t> game_t;
  typedef population_manager<agent_t> popmanager_t;

  arena<game_t, random_tournament, popmanager_t> a(tpg, ppt);
  a.evolution(threads, ngames);

  return 0;
}
