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

struct test_vars {
  double wins;
  double ties;
  double speed;
  test_vars() {
    wins = 0;
    ties = 0;
    speed = 0;
  }
};

test_vars test_games(game_generator_ptr ggn, agent_ptr a, agent_ptr b) {
  int ntest = 10;
  test_vars res;
  MutexType m;

  for (int j = 0; j < ntest; j++) {
    game_ptr gr = ggn->generate_starting_state(ggn->make_teams({a, b}));
    gr->play(1);

    int pid = gr->team_clone_ids(a->team).front();

    m.Lock();
    res.wins += a->team == gr->winner;
    res.ties += gr->winner == -1;
    res.speed += gr->score_simple(pid);
    m.Unlock();

    // ss << epoch << comma << "refbot" << comma << a->rank << comma << a->status_report() << comma << gr->end_stats() << endl;
  }

  res.wins /= ntest;
  res.ties /= ntest;
  res.speed /= ntest;

  return res;
}

void run_tests(game_generator_ptr ggn, population_manager_ptr pm) {
  cout << "Start running tests" << endl;

  agent_ptr ref = pm->refbot;
  if (!ref) ref = ggn->refbot_generator();

#pragma omp parallel for
  for (int i = 0; i < pm->pop.size(); i++) {
    agent_ptr a = pm->pop[i];
    test_vars res = test_games(ggn, a, ref);

    a->score_refbot.push(res.wins + 0.5 * res.ties);
    a->score_simple.push(res.speed);
  }

  cout << "Completed running tests" << endl;
}

void write_stats(unsigned int run_id, unsigned int epoch, game_generator_ptr ggn, population_manager_ptr pop) {
  pop->sortpop();

  // player stats
  ofstream fstat("data/run-" + to_string(run_id) + "-population.csv", ios::app);
  fstat << pop->pop_stats(to_string(epoch));
  fstat.close();

  vector<agent_ptr> buf = pop->topn(3);
  for (int i = 0; i < 3; i++) {
    agent_ptr a = buf[i];
    ofstream f("brains/run-" + to_string(run_id) + ".e" + to_string(epoch) + "p" + to_string(i));
    f << serialize_agent(a);
    f.close();
  }
}

void evolution(game_generator_ptr ggn, tournament_ptr trm, population_manager_ptr pop, int threads, string loadfile) {
#ifndef DEBUG
  omp_set_num_threads(threads);
#endif

  unsigned int start_epoch = 1;
  unsigned int run_id = rand_int(1, INT32_MAX);
  bool did_load = false;

  if (loadfile.length() > 0) {
    ifstream f(loadfile, ios::in);
    stringstream ss;
    ss << f.rdbuf();

    ss >> run_id >> start_epoch;
    pop->deserialize(ss);
    did_load = true;
    cout << "Arena: loaded epoch " << start_epoch << endl;
  }

  for (unsigned int epoch = start_epoch; true; epoch++) {
    if (!did_load) {
      cout << "ARENA: RUN ID: " << run_id << ": starting epoch " << epoch << endl;
      pop->prepare_epoch(epoch, ggn);

      run_tests(ggn, pop);
      trm->run(pop, ggn, epoch);
      run_tests(ggn, pop);

      cout << "Arena: epoch " << epoch << ": completed game rounds, generating epoch stats" << endl;
      write_stats(run_id, epoch, ggn, pop);
      cout << "Done" << endl;
    }

    // train on all games and update player scores
    cout << "Start mating and mutating" << endl;
    pop->evolve(ggn);
    cout << "Mating and mutation done" << endl;
    did_load = false;
  }
}
