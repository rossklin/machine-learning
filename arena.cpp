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
#include "random_tournament.hpp"
#include "tournament.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

#define MATING_SCORE_SINK 0.95
#define MUTATION_SCORE_SINK 0.95
#define SCORE_UPDATE_RATE 0.01

void write_stats(int epoch, game_generator_ptr ggn, population_manager_ptr pop) {
  // backup complete population
  ofstream f("save/autosave-epoch-" + to_string(epoch) + ".txt");
  f << epoch << " " << pop->serialize();
  f.close();

  // player stats
  ofstream fstat("data/population.csv", ios::app);
  fstat << pop->pop_stats(to_string(epoch));
  fstat.close();

  // reference game
  string xmeta;
  for (int j = 0; j < 3; j++) {
    game_ptr gr = ggn->team_bots_vs(pop->pop[0]);
    gr->enable_output = true;
    gr->play(epoch);
    ofstream fmeta("data/game.meta.csv", ios::app);

    xmeta = gr->end_stats();
    string comma = ",";
    fmeta << epoch << comma << pop->pop[0]->id << comma << xmeta << endl;
    fmeta.close();
  }

  cout << "Completed epoch " << epoch << ", meta: " << endl
       << xmeta << endl;
}

void evolution(game_generator_ptr ggn, tournament_ptr trm, population_manager_ptr pop, int threads, string loadfile) {
#ifndef DEBUG
  omp_set_num_threads(threads);
#endif

  int start_epoch = 1;

  if (loadfile.length() > 0) {
    ifstream f(loadfile, ios::in);
    stringstream ss;
    ss << f.rdbuf();

    ss >> start_epoch;
    pop->deserialize(ss);
    start_epoch++;
  }

  for (int epoch = start_epoch; true; epoch++) {
    pop->prepare_epoch(epoch, ggn);
    trm->run(pop, ggn, epoch);

    // train on all games and update player scores
    cout << "Arena: epoch " << epoch << ": completed game rounds" << endl;
    pop->evolve(ggn);

    cout << "Mating and mutation done, generating epoch stats" << endl;
    write_stats(epoch, ggn, pop);
  }
}
