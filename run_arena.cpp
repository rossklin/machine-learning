#include <cstring>

#include "arena.hpp"
#include "evaluator.hpp"
#include "game_generator.hpp"
#include "pod_game.hpp"
#include "pod_game_generator.hpp"
#include "population_manager.hpp"
#include "random_tournament.hpp"
#include "simple_pod_evaluator.hpp"

using namespace std;

agent_ptr agent_gen() {
  agent_ptr a(new pod_agent);
  a->eval = evaluator_ptr(new tree_evaluator);
  a->label = "tree-pod-agent";
  return a;
}

agent_ptr refbot_gen() {
  agent_ptr a(new pod_agent);
  a->eval = evaluator_ptr(new simple_pod_evaluator);
  a->label = "simple-pod-agent";
  return a;
}

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

  game_generator_ptr ggen(new pod_game_generator(tpg, ppt, agent_gen, refbot_gen));

  arena<random_tournament, default_population_manager> a(ggen);
  a.evolution(threads, ngames);

  return 0;
}
