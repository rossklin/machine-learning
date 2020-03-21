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

agent_ptr agent_gen() {
  agent_ptr a(new pod_agent);
  int depth = 4;
  a->eval = evaluator_ptr(new tree_evaluator(depth));
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

  pod_game_generator ggen(2, 1, agent_gen, refbot_gen);
  agent_ptr a = agent_gen();
  a->initialize_from_input(ggen.generate_input_sampler(), ggen.choice_dim());

  omp_lock_t writelock;
  omp_init_lock(&writelock);

  for (int epoch = 1; true; epoch++) {
    cout << "Pure train: epoch " << epoch << endl;
    a->set_exploration_rate(0.7 - 0.6 * atan(epoch / (float)40) / (M_PI / 2));
    game_ptr g = ggen.team_bots_vs(a);
    cout << "Generated game" << endl;
    auto res = g->play(epoch);
    cout << "Played game" << endl;
    for (auto pid : g->team_pids(a->team)) a->train(res.at(pid));
    a->age++;

    omp_set_lock(&writelock);
    ofstream fmeta("pure_train.meta.csv", ios::app);
    string xmeta = g->end_stats();
    fmeta << xmeta << endl;
    fmeta.close();
    omp_unset_lock(&writelock);
    cout << "Completed epoch " << epoch << endl;
  }

  omp_destroy_lock(&writelock);
}

int main() {
  pure_train();
  return 0;
}
