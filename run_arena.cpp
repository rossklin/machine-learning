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
#include "team_evaluator.hpp"
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
  bool debug = false;
  int prep_npar = 6;
  float preplim = 0.1;
  int max_turns = 300;
  int game_rounds = 100;
  int max_comp = 800;
  string loadfile;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "debug")) {
      debug = true;
      threads = 1;
      ngames = 4;
      preplim = -1;
      prep_npar = threads;
      max_turns = 20;
      game_rounds = 2;
    } else if (!strcmp(argv[i], "quick")) {
      ngames = 4;
      ppt = 1;
      preplim = 0.01;
      prep_npar = threads;
      max_turns = 100;
      game_rounds = 4;
      max_comp = 400;
    } else if (!strcmp(argv[i], "threads")) {
      threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "ngames")) {
      ngames = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "load")) {
      loadfile = argv[++i];
    } else if (!strcmp(argv[i], "preplim")) {
      preplim = atof(argv[++i]);
    } else if (!strcmp(argv[i], "ppt")) {
      ppt = atoi(argv[++i]);
    }
  }

  // todo: validate ppt matches load file

  game_generator_ptr ggen(new pod_game_generator(tpg, ppt, refbot_gen));
  ggen->prep_npar = prep_npar;
  ggen->max_turns = max_turns;
  ggen->max_complexity = max_comp;
  set<int> ireq = ggen->required_inputs();

  input_sampler is = ggen->generate_input_sampler();
  int cdim = ggen->choice_dim();

  agent_f agent_gen = [is, ppt, cdim, ireq]() {
    agent_ptr a(new pod_agent);
    vector<evaluator_ptr> evals;
    for (int i = 0; i < ppt; i++) evals.push_back(tree_evaluator::ptr(new tree_evaluator));
    a->eval = team_evaluator::ptr(new team_evaluator(evals, 4));
    a->label = "tree-pod";
    a->initialize_from_input(is, cdim, ireq);
    return a;
  };

  tournament_ptr t(new random_tournament(game_rounds));

  int popsize = ngames * tpg;
  population_manager_ptr p(new default_population_manager(popsize, agent_gen, preplim));

  evolution(ggen, t, p, threads, loadfile);

  return 0;
}
