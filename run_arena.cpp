#include <cstring>

#include "arena.hpp"
#include "evaluator.hpp"
#include "game_generator.hpp"
#include "pod_game.hpp"
#include "pod_game_generator.hpp"
#include "population_manager.hpp"
#include "random_tournament.hpp"
#include "simple_pod_evaluator.hpp"
#include "tree_evaluator.hpp"

using namespace std;

agent_ptr refbot_gen() {
  agent_ptr a(new pod_agent);
  a->eval = evaluator_ptr(new simple_pod_evaluator);
  a->label = "simple-pod-agent";
  return a;
}

int main(int argc, char **argv) {
  int threads = 6;
  int ngames = 16;
  int ppt = 2;
  int tpg = 2;
  int tree_depth = 5;
  float preplim = 0.5;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "debug")) {
      threads = 1;
      tree_depth = 5;
      ngames = 4;
      preplim = 0.01;
    } else if (!strcmp(argv[i], "quick")) {
      tree_depth = 5;
      ngames = 4;
      ppt = 1;
      preplim = 0.2;
    } else if (!strcmp(argv[i], "threads")) {
      threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "ngames")) {
      ngames = atoi(argv[++i]);
    }
  }

  game_generator_ptr ggen(new pod_game_generator(tpg, ppt, refbot_gen));
  input_sampler is = ggen->generate_input_sampler();
  int cdim = ggen->choice_dim();

  agent_f agent_gen = [is, cdim, tree_depth]() {
    agent_ptr a(new pod_agent);
    a->eval = evaluator_ptr(new tree_evaluator(tree_depth));
    a->label = "tree-pod-agent";
    a->initialize_from_input(is, cdim);
    return a;
  };

  tournament_ptr t(new random_tournament);

  int popsize = ngames * tpg;
  population_manager_ptr p(new default_population_manager(popsize, agent_gen, preplim));

  evolution(ggen, t, p);

  return 0;
}
