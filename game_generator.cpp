
#include "game_generator.hpp"

#include <algorithm>
#include <cassert>
#include <set>

#include "agent.hpp"
#include "game.hpp"
#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

game_generator::game_generator(int teams, int ppt, agent_f refbot_generator) : nr_of_teams(teams), ppt(ppt), refbot_generator(refbot_generator) {
  prep_npar = 32;
  max_turns = 100;
}

// vector<agent_ptr> game_generator::make_teams(vector<agent_ptr> ps) const {
//   vector<agent_ptr> buf(nr_of_teams * ppt);
//   set<int> pids;

//   for (int tid = 0; tid < ps.size(); tid++) {
//     agent_ptr b = ps[tid];
//     b->team = tid;
//     for (int k = 0; k < ppt; k++) {
//       agent_ptr a = b->clone();
//       a->team = tid;
//       a->team_index = k;
//       buf[tid * ppt + k] = a;
//       pids.insert(a->id);
//     }
//   }

//   assert(pids.size() == nr_of_teams * ppt);

//   return buf;
// }

team game_generator::clone_team(agent_ptr a) const {
  return team(replicate<agent_ptr>([a]() -> agent_ptr { return a->clone(); }, ppt));
}

game_ptr game_generator::team_bots_vs(team t) const {
  vector<team> buf;

  buf.push_back(t);
  for (int i = 1; i < nr_of_teams; i++) buf.push_back(clone_team(refbot_generator()));

  return generate_starting_state(buf);
}

int game_generator::choice_dim() const {
  game_ptr g = team_bots_vs(clone_team(refbot_generator()));
  agent_ptr p = g->team_table.begin()->second.players.front();
  choice_ptr c = p->select_choice(g);
  vec input = g->vectorize_input(c, p->id);
  return input.size();
}

function<vec()> game_generator::generate_input_sampler() const {
  cout << "Generate input sampler: start" << endl;
  game_ptr g = team_bots_vs(clone_team(refbot_generator()));
  cout << "Generate input sampler: play sample game" << endl;
  auto buf = g->play(0);
  cout << "Generate input sampler: complete" << endl;

  return [buf]() -> vec {
    vector<int> keys = hm_keys(buf);
    int pid = sample_one(keys);
    vector<record> recs = buf.at(pid);
    return sample_one(recs).input;
  };
}

// Let #npar bots play 100 games, select those which pass score limit and then select the best one
agent_ptr game_generator::prepared_player(input_sampler isam, agent_f gen, float plim) const {
  vector<agent_ptr> buf(prep_npar);

  auto vgen = [this, gen]() -> agent_ptr {
    agent_ptr a = gen();
    while (set_difference(required_inputs(), a->eval->list_inputs()).size() > 0) {
      a = gen();
    }
    return a;
  };

#pragma omp parallel for
  for (int t = 0; t < prep_npar; t++) {
    float eval = 0;
    agent_ptr a = vgen();
    int restarts = 0;
    int max_its = 5;
    int ndata = 0;

    for (int i = 0; i < max_its; i++) {
      bool supervizion = i % 2 == 0 && i < 4;
      a->set_exploration_rate(0.8 - 0.6 * i / (float)max_its);

      // play and train
      vector<team> ts;
      if (supervizion) {
        // prepare teams for supervized learning
        if (a->tutor_buf.size() > 0) {
          int c = 0, n = a->tutor_buf.size();
          random_shuffle(a->tutor_buf.begin(), a->tutor_buf.end());
          while (ts.size() < nr_of_teams) ts.push_back(a->tutor_buf[c++ % n]);
        } else {
          ts = replicate<team>([this]() { return clone_team(refbot_generator()); }, nr_of_teams);
        }

      } else {
        // prepare teams for self practice
        ts = replicate<team>([this, a]() { return clone_team(a); }, nr_of_teams);
      }

      game_ptr g = generate_starting_state(ts);
      const auto res = g->play(i + 1);
      for (auto x : res) a->train(x.second, isam);

      bool complex = a->eval->complexity() > 5;
      bool stable = a->eval->stable;
      bool trainable = a->tstats.rate_successfull > pow(0.975, max_its);

      if (!(complex && stable && trainable)) {
        // this agent has degenerated
        i = 0;
        a = vgen();
        eval = 0;
        restarts++;
        continue;
      }

      if (!supervizion) {
        for (auto tid : hm_keys(g->team_table)) {
          for (auto a : g->team_table.at(tid).players) {
            double res = g->score_simple(a->id);
            ndata += i;
            eval = (res * i + (ndata - i) * eval) / ndata;
          }
        }
      }
    }

    if (eval > plim) {
      a->simple_score = eval;
      buf[t] = a;
      cout << "prepared_player (" << restarts << " restarts): ACCEPTING " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
    } else {
      cout << "prepared_player (" << restarts << " restarts): rejecting " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
    }
  }

  sort(buf.begin(), buf.end(), [](agent_ptr a, agent_ptr b) -> bool {
    double sa = a ? a->simple_score : 0;
    double sb = b ? b->simple_score : 0;
    return sa > sb;
  });

  return buf[0];
};

// Repeatedly attempt to make prepared players until n have been generated
vector<team> game_generator::prepare_n(agent_f gen, int n, float plim) const {
  vector<team> buf;
  input_sampler isam = generate_input_sampler();

  for (int i = 0; i < n; i++) {
    cout << "prepare_n: starting " << (i + 1) << "/" << n << endl
         << "----------------------------------------" << endl;
    vector<agent_ptr> pl = replicate<agent_ptr>(
        [this, isam, gen, plim]() {
          agent_ptr a = 0;
          while (!a) a = prepared_player(isam, gen, plim);
          return a;
        },
        ppt);

    buf.push_back(team(pl));
    cout << "prepare_n: completed " << (i + 1) << "/" << n << " score " << buf[i].score << endl
         << "----------------------------------------" << endl;
  }

  return buf;
}
