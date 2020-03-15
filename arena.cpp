#include <omp.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "agent.hpp"
#include "arena.hpp"
#include "game.hpp"
#include "game_generator.hpp"
#include "population_manager.hpp"
#include "tournament.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

#define MATING_SCORE_SINK 0.95
#define MUTATION_SCORE_SINK 0.95
#define SCORE_UPDATE_RATE 0.01

// ARENA
template <typename T, typename P>
arena<T, P>::arena(game_generator_ptr ggen) : ggn(ggen) {}

// // for online learning, unlikely we need it
// void arena<G,T,P>::train_agents_sarsa(vector<choice::record_table> results) {
//   int n = results.size();
//   double alpha = 0.1;
//   double gamma = 0.8;

//   // local training
//   for (auto x : players) {
//     for (int i = 0; i < n - 1; i++) {
//       choice::record y1 = results[i][x.first];
//       choice::record y2 = results[i+1][x.first];
//       double q1 = y1.output;
//       double q2 = y2.output;
//       double r = y1.reward;
//       x.second->cval->update(y1.input, q1 + alpha * (r + gamma * q2 - q1));
//     }
//   }
// }

template <typename T, typename P>
void arena<T, P>::evolution(int threads, int ngames) {
#ifndef DEBUG
  omp_set_num_threads(threads);
#endif

  for (int epoch = 1; true; epoch++) {
    pop.prepare_epoch(epoch, ggn);
    trm.run(pop);

    // train on all games and update player scores
    cout << "Arena: epoch " << epoch << ": completed game rounds" << endl;
    pop->evolve();

    cout << "Mating and mutation done, generating epoch stats" << endl;
    write_stats(epoch);
  }
}

template <typename T, typename P>
void arena<T, P>::write_stats(int epoch) const {
  // store best brains
  auto player = pop->topn(3);
  for (int i = 0; i < 3 && i < player.size(); i++) {
    ofstream f("brains/p" + to_string(i) + "_r" + to_string(epoch) + ".txt");
    f << player[i]->serialize();
    f.close();
  }

  // player stats
  ofstream fstat("population.csv", ios::app);
  fstat << pop->pop_stats(to_string(epoch));
  fstat.close();

  // reference game
  string xmeta;
  for (int j = 0; j < 3; j++) {
    game_ptr gr = ggn->team_bots_vs(player[0]);
    gr->enable_output = true;
    gr->play(epoch);
    ofstream fmeta("game.meta.csv", ios::app);

    // find opponent to print
    agent_ptr op;
    for (auto p : gr->players) {
      if (p.second->team != 0) {
        op = p.second;
        break;
      }
    }

    xmeta = gr->end_stats(player[0]->id, op->id);
    fmeta << xmeta << endl;
    fmeta.close();
  }

  cout << "Completed epoch " << epoch << ", meta: " << endl
       << xmeta << endl;
}