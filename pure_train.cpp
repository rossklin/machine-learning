#include <omp.h>
#include <cmath>
#include <fstream>

#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;

void pure_train() {
  int tree_depth = 10;
  pod_game_ptr base_game(new pod_game(2, 1, tree_depth));
  input_sampler isampler = base_game->generate_input_sampler();
  int choice_dim = base_game->choice_dim();

  // initialize players
  int n = 100;
  vector<agent_ptr> pool(n);
  for (int i = 0; i < n; i++) {
    pool[i] = base_game->generate_player();
    pool[i]->initialize_from_input(isampler, choice_dim);
  }

  omp_lock_t writelock;
  omp_init_lock(&writelock);

  for (int epoch = 1; true; epoch++) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      vector<agent_ptr> p(2);
      p[0] = pool[i];
      p[0]->set_exploration_rate(0.7 - 0.6 * atan(epoch / (float)40) / (M_PI / 2));
      p[1] = base_game->generate_refbot();

      pod_game_ptr g = static_pointer_cast<pod_game>(base_game->generate_starting_state(p));
      g->play();

      g->train(p[0]->id, -1, p[0]->team);
      p[0]->age++;

      omp_set_lock(&writelock);
      ofstream fmeta("pure_train.meta.csv", ios::app);
      string xmeta = g->end_stats(p[0]->id, p[1]->id);
      fmeta << xmeta << endl;
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
