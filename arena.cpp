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
#include "types.hpp"
#include "utility.hpp"

using namespace std;

#define MATING_SCORE_SINK 0.95
#define MUTATION_SCORE_SINK 0.95
#define SCORE_UPDATE_RATE 0.01

// ARENA

arena::arena(game_ptr bg, int threads, int ppt, int tpg, float plim) : base_game(bg) {
  assert(threads > 0);
  assert(ppt > 0);
  assert(tpg > 1);

  this->threads = threads;
  this->ppt = ppt;
  this->tpg = tpg;
  this->ppg = ppt * tpg;
  this->preplim = plim;
}

// // for online learning, unlikely we need it
// void arena::train_agents_sarsa(vector<choice::record_table> results) {
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

double mate_score(agent_ptr parent1, agent_ptr x) {
  // x is protected so score is not reliable
  if (x->was_protected) return 0;

  // direct descendant
  if (parent1->parents.count(x->id) || x->parents.count(parent1->id)) return 0;

  double different_ancestors = set_symdiff(x->ancestors, parent1->ancestors).size();
  double common_parents = set_intersect(x->parents, parent1->parents).size();

  return x->score + different_ancestors - common_parents;
};

// Let #threads bots play 100 games, select those which pass score limit and then select the best one
agent_ptr arena::prepared_player(function<agent_ptr()> gen, float plim) {
  vector<agent_ptr> buf(threads);

#pragma omp parallel for
  for (int t = 0; t < threads; t++) {
    float eval = 0;
    agent_ptr a;
    for (int j = 0; j < 10000 && !a; j++) a = gen();
    assert(a);

    for (int i = 0; i < 100; i++) {
      a->set_exploration_rate(0.5 - 0.4 * i / (float)100);

      // play and train
      game_ptr g = base_game->team_bots_vs(a);
      g->play();
      g->train(a->id, -1, -1);
      eval = 0.1 * g->score_simple(a->id) + 0.9 * eval;

      if (i > 20 && eval < 0.001 * (float)i) break;
    }

    if (eval > plim) {
      a->score = eval;
      buf[t] = a;
    }
  }

  sort(buf.begin(), buf.end(), [](agent_ptr a, agent_ptr b) -> bool {
    double sa = a ? a->score : 0;
    double sb = b ? b->score : 0;
    return sa > sb;
  });

  return buf[0];
};

// Repeatedly attempt to make prepared players until n have been generated

vector<agent_ptr> arena::prepare_n(function<agent_ptr()> gen, int n, float plim) {
  vector<agent_ptr> buf(n);

  for (int i = 0; i < n; i++) {
    cout << "prepare_n: starting " << (i + 1) << "/" << n << endl;
    agent_ptr
        a = 0;
    while (!a) a = prepared_player(gen, plim);
    buf[i] = a;
    cout << "prepare_n: completed " << (i + 1) << "/" << n << endl;
  }

  return buf;
}

void arena::evolution(int ngames) {
#ifndef DEBUG
  omp_set_num_threads(threads);
#endif

  // generate an initialized agent

  vector<agent_ptr> player, player_buf, player_buf2, inner_player(ppg);
  vector<game_ptr> game_record;
  int i;
  int base_population = tpg * ngames;
  int game_rounds = 20;
  int practice_rounds = 3;
  int nkeep = fmax(0.5 * base_population, 1);

  for (int epoch = 1; true; epoch++) {
    double ltime = log(epoch + 100);
    float winner_reward = base_game->winner_reward(epoch);
    int protected_age = ltime / 2 + 1;
    int protected_mut_age = ltime / 5 + 1;

    // fill remaining population with new samples
    int n_fill = base_population - player.size();
    cout << "Arena: starting epoch: " << epoch << ": generating " << n_fill << " new players." << endl;

    if (n_fill > 0) {
      player_buf = prepare_n([this]() { return base_game->generate_player(); }, n_fill, preplim);
      player.insert(player.end(), player_buf.begin(), player_buf.end());
    }

    if (player.size() > base_population) {
      player.resize(base_population);
    }

    // set agent selector randomness and score
    double q = 0.5 + 0.4 * sigmoid(epoch, 300);
    for (auto a : player) {
      a->last_score = a->score;
      a->set_exploration_rate(q);
    }

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

    // train on all games and update player scores
    cout << "Arena: epoch " << epoch << ": completed game rounds" << endl;

    // select nkeep best players
    sort(player.begin(), player.end(), [](agent_ptr a, agent_ptr b) -> bool { return a->score > b->score; });

    // update simple score limit
    int lim_idx = 0.67 * player.size();
    simple_score_limit = 0.1 * player[lim_idx]->simple_score + 0.9 * simple_score_limit;

    player_buf = player;
    player_buf.resize(nkeep);

    // protect children and players who's scores are still increasing
    int n_protprog = 0;
    int n_protchild = 0;
    int n_protmut = 0;
    for (int i = nkeep; i < player.size(); i++) {
      agent_ptr a = player[i];
      if (a->was_protected) player_buf.push_back(a);

      // update protection after adding agent so scores are calculated
      // before the agent is evaluated
      double delta = a->score - a->last_score;
      double limit_score = player[nkeep - 1]->score - a->score;
      bool protect_progress = delta > limit_score / 20;
      bool protect_child = a->age < protected_age;
      bool protect_mutant = a->mut_age < protected_mut_age;

      a->was_protected = protect_progress || protect_child || protect_mutant;

      n_protmut += protect_mutant;
      n_protprog += protect_progress;
      n_protchild += protect_child;
    }

    int free_spots = player.size() - player_buf.size();
    int used_spots = 0;

    cout << "Arena: keeping " << nkeep << ", protected " << n_protprog << " progressors, " << n_protmut << " mutants and " << n_protchild << " children" << endl;

    auto mate_generator = [this, nkeep, player]() -> agent_ptr {
      int idx1 = rand_int(0, nkeep - 1);
      agent_ptr parent1 = player[idx1];

      vector<agent_ptr> buf = player;
      buf.erase(buf.begin() + idx1);

      sort(buf.begin(), buf.end(), [parent1](agent_ptr a, agent_ptr b) {
        return mate_score(parent1, a) > mate_score(parent1, b);
      });

      agent_ptr parent2 = ranked_sample(buf, 0.8);

      return parent1->mate(parent2);
    };

    auto mutate_generator = [this, nkeep, player]() -> agent_ptr {
      int idx1 = rand_int(0, nkeep - 1);
      return player[idx1]->mutate();
    };

    // mating
    int n_mate = ceil(0.5 * (float)free_spots);
    int n_mutate = ceil(0.9 * (float)(free_spots - n_mate));
    int n_init = free_spots - n_mate - n_mutate;
    vector<agent_ptr> buf;

    cout << "Mating: " << n_mate << endl;
    buf = prepare_n(mate_generator, n_mate, fmax(preplim, simple_score_limit));
    player_buf.insert(player_buf.end(), buf.begin(), buf.end());

    cout << "Mutating: " << n_mutate << endl;
    buf = prepare_n(mutate_generator, n_mutate, fmax(preplim, simple_score_limit));
    player_buf.insert(player_buf.end(), buf.begin(), buf.end());

    cout << "Mating and mutation done, generating epoch stats" << endl;

    player = player_buf;

    // store best brains
    for (i = 0; i < 3 && i < player.size(); i++) {
      ofstream f("brains/p" + to_string(i) + "_r" + to_string(epoch) + ".txt");
      f << player[i]->serialize();
      f.close();
    }

    // player stats
    ofstream fstat("population.csv", ios::app);
    string sep = ",";
    for (i = 0; i < player.size(); i++) {
      fstat << epoch << sep << player[i]->id << sep << (i + 1) << sep << player[i]->ancestors.size() << sep << player[i]->parents.size() << sep << player[i]->score << sep << player[i]->age << sep << player[i]->status_report() << endl;
    }
    fstat.close();

    // reference game
    string xmeta;
    for (int j = 0; j < 3; j++) {
      game_ptr gr = base_game->team_bots_vs(player[0]);
      gr->enable_output = true;
      gr->play();
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

    // age players
    for (auto x : player) {
      x->age++;
      x->mut_age++;
      x->score *= (1 - SCORE_UPDATE_RATE);
    }

    cout << "Completed epoch " << epoch << ", meta: " << endl
         << xmeta << endl;
  }
}
