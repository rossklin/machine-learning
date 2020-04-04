#include "random_tournament.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "agent.hpp"
#include "game.hpp"
#include "game_generator.hpp"
#include "population_manager.hpp"

using namespace std;

void random_tournament::run(population_manager_ptr pm, game_generator_ptr gg, int epoch) {
  int game_rounds = 50;
  int practice_rounds = 20;
  float score_update_rate = 0.05;
  int ppt = gg->ppt;
  int tpg = gg->nr_of_teams;
  int ngames = pm->popsize / tpg;  // one agent is used to generate each team

  pm->check_gg(gg);

  for (int round = 0; round < game_rounds; round++) {
    // play a number of games, reusing players as needed
    cout << "Arena: epoch " << epoch << " round " << round << ": select players" << endl;

    bool practice = round < practice_rounds;
    float use_exrate = practice ? 0.4 : 0.05;

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

    cout << "Arena: epoch " << epoch << " round " << round << ": play games" << endl;

#pragma omp parallel for
    for (int i = 0; i < game_record.size(); i++) game_record[i]->result_buf = game_record[i]->play(epoch);

    // train on all games and update player scores
    cout << "Arena: epoch " << epoch << " round " << round << ": start training" << endl;

#pragma omp parallel for
    for (int i = 0; i < pm->popsize; i++) {
      agent_ptr a = pm->pop[i];
      game_ptr g = game_record[a->assigned_game];
      for (auto pid : g->team_clone_ids(a->team)) a->train(g->result_buf.at(pid));
    }

    // do not update scores if this is a practice round
    if (practice) continue;

    // update scores
    for (int i = 0; i < pm->pop.size(); i++) {
      agent_ptr p = pm->pop[i];
      game_ptr g = game_record[p->assigned_game];
      vector<int> clone_ids = g->team_clone_ids(p->team);

      // update simple score
      double a = score_update_rate;
      for (auto pid : clone_ids) p->simple_score = a * g->score_simple(pid) + (1 - a) * p->simple_score;

      // force game to select a winner by heuristic if the game was a tie
      if (g->winner == -1) g->select_winner();

      // not possible to determine winner, no score updating
      if (g->winner == -1) continue;

      double score_defeated = 0;
      double score_winner = 0;
      for (auto x : g->original_agents) {
        if (x->team != g->winner) score_defeated += x->score;
      }

      for (auto x : g->original_agents) {
        if (x->team == g->winner) score_winner += x->score;
      }

      score_defeated /= ppt * (tpg - 1);
      score_winner /= ppt;

      if (p->team == g->winner) {
        if (p->score >= score_defeated) {
          p->score += a;
        } else {
          p->score += a * (score_defeated + 1);
        }
      } else {
        if (p->score < score_winner) {
          p->score = (1 - a) * p->score;
        } else {
          p->score = (1 - a) * (0.9 * p->score + 0.1 * score_defeated);
        }
      }
    }
  }
}