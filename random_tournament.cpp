#include "random_tournament.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "agent.hpp"
#include "game.hpp"
#include "game_generator.hpp"
#include "population_manager.hpp"
#include "utility.hpp"

using namespace std;

random_tournament::random_tournament(int gr) : tournament(), game_rounds(gr) {}

void random_tournament::run(population_manager_ptr pm, game_generator_ptr gg, int epoch) {
  int practice_rounds = 0.3 * game_rounds;
  float score_update_rate = 0.1;
  int ppt = gg->ppt;
  int tpg = gg->nr_of_teams;
  int ngames = pm->popsize / tpg;  // one agent is used to generate each team
  input_sampler isam = gg->generate_input_sampler();
  vector<vector<vector<record>>> training_data(pm->pop.size());

  pm->check_gg(gg);

  for (auto a : pm->pop) a->score_tmt_buf = a->score_tmt.current;
  int batch_size = 10;
  for (int round = 0; round < game_rounds; round++) {
    // play a number of games, reusing players as needed
    cout << "Epoch " << epoch << ": RT games round " << round << " of " << game_rounds << "     \r";

    bool practice = round < practice_rounds;
    float use_exrate = practice ? 0.5 : 0.05;

    // match players in games
    auto player_buf = pm->pop;
    random_shuffle(player_buf.begin(), player_buf.end());
    for (auto a : player_buf) a->set_exploration_rate(use_exrate);

    vector<game_ptr> game_record(ngames);

    // agents play vs each other
    for (int idx = 0; idx < ngames; idx++) {
      vector<agent_ptr> assign_players(player_buf.begin() + idx * tpg, player_buf.begin() + (idx + 1) * tpg);
      for (auto a : assign_players) a->assigned_game = idx;

      // this is a bit ugly: make_teams sets the team of the original agent as well as the clone
      game_record[idx] = gg->generate_starting_state(gg->make_teams(assign_players));
      game_record[idx]->original_agents = assign_players;
    }

#pragma omp parallel for
    for (int i = 0; i < game_record.size(); i++) game_record[i]->result_buf = game_record[i]->play(epoch);

    // update scores
    for (int i = 0; i < pm->pop.size(); i++) {
      agent_ptr p = pm->pop[i];
      game_ptr g = game_record[p->assigned_game];
      vector<int> clone_ids = g->team_clone_ids(p->team);

      // update simple score
      double a = score_update_rate / game_rounds;  // 1e-3
      // for (auto pid : clone_ids) p->simple_score = a * g->score_simple(pid) + (1 - a) * p->simple_score;

      // force game to select a winner by heuristic if the game was a tie
      if (g->winner == -1) g->select_winner();

      // // not possible to determine winner, no score updating
      // if (g->winner == -1) continue;

      double score_defeated = 0;
      double score_winner = 0;
      for (auto x : g->original_agents) {
        if (x->team != g->winner) score_defeated += x->score_tmt_buf;
      }

      for (auto x : g->original_agents) {
        if (x->team == g->winner) score_winner += x->score_tmt_buf;
      }

      score_defeated /= ppt * (tpg - 1);
      score_winner /= ppt;

      bool expected = score_winner >= score_defeated;
      bool win = p->team == g->winner;
      int sig = win - !win;
      double diff = sig * a;
      if (!expected) diff *= (score_defeated - score_winner + 1);

      // add training data for all clones of agent
      for (auto pid : clone_ids) {
        training_data[i].push_back(g->result_buf[pid]);
      }

      // if agent lost, also add training data for opponent clones
      if (p->age < p->inspiration_age_limit && !win) {
        for (auto pid : g->team_clone_ids(g->winner)) {
          training_data[i].push_back(g->result_buf[pid]);
        }
      }

      if (!practice) p->score_tmt_buf += diff;
    }

    // Run training batch
    if (round % batch_size == 0 || round == game_rounds - 1) {
      cout << endl
           << "RT: training batch" << endl;
#pragma omp parallel for
      for (int i = 0; i < pm->pop.size(); i++) {
        pm->pop[i]->train(training_data[i], isam);
      }
      training_data = vector<vector<vector<record>>>(pm->pop.size());
    }
  }

  for (auto a : pm->pop) a->score_tmt.push(a->score_tmt_buf);
  pm->sortpop();

  cout << "RT: complete" << endl;
}