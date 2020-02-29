#include <iostream>

#include "random_tournament.hpp"

using namespace std;

void random_tournament::run(population_manager_ptr pm, int epoch) {
  int game_rounds = 10;
  int practice_rounds = 3;
  for (int round = 0; round < game_rounds; round++) {
    // play a number of games, reusing players as needed
    cout << "Arena: epoch " << epoch << " round " << round << (round < practice_rounds ? " (practice)" : "") << ": select players" << endl;

    // match players in games
    player_buf = player;
    game_record.clear();

    // agents play vs each other
    while (player_buf.size()) {
      vector<agent_ptr> assign_players;
      int game_idx = game_record.size();
      assign_players.clear();

      // fill other teams with other players
      for (int j = 0; j < tpg; j++) {
        // attempt to select player with proper age
        if (player_buf.size()) {
          int idx = rand_int(0, player_buf.size() - 1);
          agent_ptr b = player_buf[idx];
          b->assigned_game = game_idx;
          player_buf.erase(player_buf.begin() + idx);
          assign_players.push_back(b);
        } else {
          assign_players.push_back(base_game->generate_refbot());
        }
      }

      inner_player = base_game->make_teams(assign_players);
      assert(inner_player.size() == ppg);
      game_record.push_back(base_game->generate_starting_state(inner_player));
    }

    cout << "Arena: epoch " << epoch << " round " << round << ": play games" << endl;

#pragma omp parallel for
    for (int i = 0; i < game_record.size(); i++) game_record[i]->play();

    // train on all games and update player scores
    cout << "Arena: epoch " << epoch << " round " << round << ": start training" << endl;

#pragma omp parallel for
    for (i = 0; i < player.size(); i++) {
      agent_ptr p = player[i];
      if (p->assigned_game > -1) {
        game_record[p->assigned_game]->train(p->id, -1, p->team);
      } else {
        throw runtime_error("A player was not assigned to a game!");
      }
    }

    // remove unstable players
    for (i = 0; i < player.size(); i++) {
      if (!player[i]->evaluator_stability()) {
        cout << "Erasing unstable player " << player[i]->id << endl;
        player.erase(player.begin() + i--);
      }
    }

    // update scores
    for (i = 0; i < player.size(); i++) {
      agent_ptr p = player[i];
      game_ptr g = game_record[p->assigned_game];

      // update simple score
      double a = SCORE_UPDATE_RATE;
      p->simple_score = a * g->score_simple(p->id) + (1 - a) * p->simple_score;

      // force game to select a winner by heuristic if the game was a tie
      if (g->winner == -1) g->select_winner();

      double score_defeated = 0;
      double score_winner = 0;
      for (auto x : g->players) {
        if (x.second->team != g->winner) score_defeated += x.second->score;
      }
      for (auto x : g->players) {
        if (x.second->team == g->winner) score_winner += x.second->score;
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