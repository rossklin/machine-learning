#include <cstring>
#include <fstream>

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
  a->label = "simple-pod";
  return a;
}

int main(int argc, char **argv) {
  int threads = 6;
  int ngames = 16;
  int ppt = 2;
  int tpg = 2;
  int tree_depth = 5;
  bool debug = false;
  int prep_npar = 32;
  float preplim = 0.5;
  int max_turns = 300;
  string loadfile;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "debug")) {
      debug = true;
      threads = 1;
      tree_depth = 3;
      ngames = 4;
      preplim = -1;
      prep_npar = threads;
      max_turns = 20;
    } else if (!strcmp(argv[i], "quick")) {
      tree_depth = 4;
      ngames = 4;
      ppt = 1;
      preplim = 0.01;
      prep_npar = threads;
      max_turns = 100;
    } else if (!strcmp(argv[i], "threads")) {
      threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "ngames")) {
      ngames = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "load")) {
      loadfile = argv[++i];
    }
  }

  // todo: validate ppt matches load file

  game_generator_ptr ggen(new pod_game_generator(tpg, ppt, refbot_gen));
  ggen->prep_npar = prep_npar;
  ggen->max_turns = max_turns;

  input_sampler is = ggen->generate_input_sampler();
  int cdim = ggen->choice_dim();

  agent_f agent_gen = [is, cdim, tree_depth]() {
    agent_ptr a(new pod_agent);
    a->eval = evaluator_ptr(new tree_evaluator(tree_depth));
    a->label = "tree-pod";
    a->initialize_from_input(is, cdim);
    return a;
  };

  tournament_ptr t(new random_tournament);

  int popsize = ngames * tpg;
  population_manager_ptr p(new default_population_manager(popsize, agent_gen, preplim));

  evolution(ggen, t, p, threads, loadfile);

  return 0;
}
