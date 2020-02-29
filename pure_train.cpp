#include <omp.h>
#include <cmath>
#include <fstream>

#include "game_generator.hpp"
#include "pod_game.hpp"
#include "population_manager.hpp"
#include "simple_pod_evaluator.hpp"
#include "utility.hpp"

using namespace std;
typedef pod_agent<simple_pod_evaluator> refbot_t;
typedef pod_agent<tree_evaluator> agent_t;
typedef game_generator<pod_game<agent_t>, refbot_t> game_t;

void pure_train() {
  game_t ggen(2, 1);
  agent_t::ptr a(new agent_t);

  omp_lock_t writelock;
  omp_init_lock(&writelock);

  for (int epoch = 1; true; epoch++) {
    a->set_exploration_rate(0.7 - 0.6 * atan(epoch / (float)40) / (M_PI / 2));
    game_t::game_ptr g = ggen.team_bots_vs(a);
    auto res = g->play(epoch);
    a->train(res.at(a->id));
    a->age++;

    omp_set_lock(&writelock);
    ofstream fmeta("pure_train.meta.csv", ios::app);
    string xmeta = g->end_stats(a->id, hm_keys(g->players).back());  // todo: get id of other player correctly
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
