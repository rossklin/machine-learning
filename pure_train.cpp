#include <omp.h>

#include <cassert>
#include <cmath>
#include <fstream>

#include "game_generator.hpp"
#include "pod_game.hpp"
#include "pod_game_generator.hpp"
#include "population_manager.hpp"
#include "simple_pod_evaluator.hpp"
#include "team_evaluator.hpp"
#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

agent_ptr agent_gen(int ppt) {
  agent_ptr a(new pod_agent);
  vector<evaluator_ptr> evals;
  for (int i = 0; i < ppt; i++) evals.push_back(tree_evaluator::ptr(new tree_evaluator));
  a->eval = team_evaluator::ptr(new team_evaluator(evals, 4));
  a->label = "tree-pod-agent";
  return a;
}

agent_ptr refbot_gen() {
  agent_ptr a(new pod_agent);
  a->eval = evaluator_ptr(new simple_pod_evaluator);
  a->label = "simple-pod-agent";
  return a;
}

void pure_train() {
  cout << "Pure train: start" << endl;
  omp_set_num_threads(6);

  pod_game_generator ggen(2, 1, refbot_gen);
  vector<agent_ptr> pop(1000);
  input_sampler isam = ggen.generate_input_sampler();
  int cdim = ggen.choice_dim();
  set<int> ireq = ggen.required_inputs();

  cout << "cdim: " << cdim << endl;

  auto vgen = [ggen, cdim, isam, ireq]() -> agent_ptr {
    agent_ptr a = agent_gen(cdim);
    a->initialize_from_input(isam, cdim, ireq);
    while (set_difference(ireq, a->eval->list_inputs()).size() > 0) {
      a = agent_gen(cdim);
      a->initialize_from_input(isam, cdim, ireq);
    }
    return a;
  };

  for (auto &a : pop) a = vgen();

  omp_lock_t writelock;
  omp_init_lock(&writelock);

  for (int epoch = 1; true; epoch++) {
    cout << "Pure train: epoch " << epoch << endl;

#pragma omp parallel for
    for (int i = 0; i < pop.size(); i++) {
      agent_ptr a = pop[i];
      if ((set_difference(ggen.required_inputs(), a->eval->list_inputs()).size() > 0)) continue;
      if (!a->eval->stable) continue;

      a->set_exploration_rate(0.7 - 0.6 * atan(epoch / (float)40) / (M_PI / 2));

      game_ptr g = ggen.team_bots_vs(a);
      assert(g->players.size() == 2);

      auto res = g->play(epoch);
      for (auto x : res) a->train(x.second, isam);

      omp_set_lock(&writelock);
      ofstream fmeta("pure_train.meta.csv", ios::app);
      string comma = ",";
      fmeta << epoch << comma << a->status_report() << comma << g->end_stats() << endl;
      fmeta.close();
      omp_unset_lock(&writelock);
    }

    cout << "Completed epoch " << epoch << endl;
  }

  omp_destroy_lock(&writelock);
}

int main() {
  pure_train();
  return 0;
}
