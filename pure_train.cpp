#include <omp.h>
#include <cmath>
#include <fstream>

#include "game_generator.hpp"
#include "pod_game.hpp"
#include "pod_game_generator.hpp"
#include "population_manager.hpp"
#include "simple_pod_evaluator.hpp"
#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

agent_ptr agent_gen(int cdim) {
  agent_ptr a(new pod_agent);
  int depth = 4;
  shared_ptr<tree_evaluator> e(new tree_evaluator(depth));
  // e->example_setup(cdim);
  a->eval = e;
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
  omp_set_num_threads(7);

  pod_game_generator ggen(2, 1, refbot_gen);
  vector<agent_ptr> pop(100);
  for (auto &a : pop) {
    a = agent_gen(ggen.choice_dim());
    a->initialize_from_input(ggen.generate_input_sampler(), ggen.choice_dim());
  }

  omp_lock_t writelock;
  omp_init_lock(&writelock);

  for (int epoch = 1; true; epoch++) {
    cout << "Pure train: epoch " << epoch << endl;

#pragma omp parallel for
    for (int i = 0; i < pop.size(); i++) {
      agent_ptr a = pop[i];
      if (!a->eval->stable) continue;

      a->set_exploration_rate(0.7 - 0.6 * atan(epoch / (float)40) / (M_PI / 2));
      game_ptr g = ggen.team_bots_vs(a);
      auto res = g->play(epoch);

      for (auto pid : g->team_pids(a->team)) a->train(res.at(pid));

      a->age++;

      omp_set_lock(&writelock);
      ofstream fmeta("pure_train.meta.csv", ios::app);
      string xmeta = g->end_stats();
      fmeta << epoch << "," << a->id << "," << xmeta << endl;
      fmeta.close();
      omp_unset_lock(&writelock);
      cout << "Completed epoch " << epoch << endl;
    }
  }

  omp_destroy_lock(&writelock);
}

int main() {
  pure_train();
  return 0;
}
