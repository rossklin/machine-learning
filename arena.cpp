#include "arena.hpp"

#include <omp.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "agent.hpp"
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
  stringstream ss;
  string row_prefix;
  for (int rank = 0; rank < 3; rank++) {
    row_prefix = to_string(epoch) + "," + to_string(rank) + ",refbot,";
    // play five refgames vs refbot
    for (int j = 0; j < 5; j++) {
      agent_ptr a = pop->pop[rank];
      game_ptr gr = ggn->team_bots_vs(a);

      gr->enable_output = true;
      gr->play(epoch, row_prefix);

      ss << epoch << comma << "refbot" << comma << a->rank << comma << a->status_report() << comma << gr->end_stats() << endl;
    }

    // play five refgames vs retired agents
    if (!pop->retirement.size()) continue;

    int ret_idx = min((int)pop->retirement.size(), 10);
    agent_ptr b = pop->retirement[ret_idx];
    row_prefix = to_string(epoch) + "," + to_string(rank) + ",retiree#" + to_string(b->id) + ",";
    for (int j = 0; j < 5; j++) {
      agent_ptr a = pop->pop[rank];
      game_ptr gr = ggn->generate_starting_state(ggn->make_teams({a, b}));

      gr->enable_output = true;
      gr->play(epoch, row_prefix);

      ss << epoch << comma << "retiree" << comma << a->rank << comma << a->status_report() << comma << gr->end_stats() << endl;
    }
  }

  string xmeta = ss.str();
  ofstream fmeta("data/game.meta.csv", ios::app);
  fmeta << xmeta;
  fmeta.close();

  vector<agent_ptr> buf = pop->topn(3);
  for (int i = 0; i < 3; i++) {
    agent_ptr a = buf[i];
    ofstream f("brains/e" + to_string(epoch) + "p" + to_string(i));
    f << serialize_agent(a);
    f.close();
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
