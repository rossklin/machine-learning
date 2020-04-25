
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

vector<agent_ptr> game_generator::make_teams(vector<agent_ptr> ps) const {
  vector<agent_ptr> buf(nr_of_teams * ppt);
  set<int> pids;

  for (int tid = 0; tid < ps.size(); tid++) {
    agent_ptr b = ps[tid];
    b->team = tid;
    for (int k = 0; k < ppt; k++) {
      agent_ptr a = b->clone();
      a->team = tid;
      a->team_index = k;
      buf[tid * ppt + k] = a;
      pids.insert(a->id);
    }
  }

  assert(pids.size() == nr_of_teams * ppt);

  return buf;
}

game_ptr game_generator::team_bots_vs(agent_ptr a) const {
  vector<agent_ptr> ps(nr_of_teams);
  ps[0] = a;
  for (int i = 1; i < nr_of_teams; i++) ps[i] = refbot_generator();
  auto buf = make_teams(ps);
  assert(buf.size() == nr_of_teams * ppt);

  return generate_starting_state(buf);
}

int game_generator::choice_dim() const {
  game_ptr g = team_bots_vs(refbot_generator());
  agent_ptr p = g->players.begin()->second;
  choice_ptr c = p->select_choice(g);
  vec input = g->vectorize_input(c, p->id);
  return input.size();
}

function<vec()> game_generator::generate_input_sampler() const {
  cout << "Generate input sampler: start" << endl;
  game_ptr g = team_bots_vs(refbot_generator());
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
agent_ptr game_generator::prepared_player(agent_f gen, float plim) const {
  vector<agent_ptr> buf(prep_npar);
  input_sampler isam = generate_input_sampler();

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
    int max_its = 30;
    int ndata = 0;

    for (int i = 0; i < max_its; i++) {
      bool supervizion = ((i * i) / (10 * max_its)) % 2 == 0;
      a->set_exploration_rate(0.8 - 0.6 * i / (float)max_its);

      // play and train
      vector<agent_ptr> pl;
      if (supervizion) {
        // prepare teams for supervized learning
        if (a->parent_buf.size() > 0) {
          int c = 0, n = a->parent_buf.size();
          random_shuffle(a->parent_buf.begin(), a->parent_buf.end());
          while (pl.size() < nr_of_teams) pl.push_back(a->parent_buf[c++ % n]);
        } else {
          pl = vector<agent_ptr>(nr_of_teams, refbot_generator());
        }

      } else {
        // prepare teams for self practice
        pl = vector<agent_ptr>(nr_of_teams, a);
      }

      game_ptr g = generate_starting_state(make_teams(pl));
      const auto res = g->play(i + 1);
      for (auto x : res) a->train(x.second, isam);

      bool complex = a->eval->complexity() > 5;
      bool stable = a->eval->stable;
      bool trainable = a->tstats.rate_successfull > 0.5;

      if (!(complex && stable && trainable)) {
        // this agent has degenerated
        i = 0;
        a = vgen();
        eval = 0;
        restarts++;
        continue;
      }

      if (!supervizion) {
        for (auto clone_pid : hm_keys(g->players)) {
          double res = g->score_simple(clone_pid);
          ndata += i;
          eval = (res * i + (ndata - i) * eval) / ndata;
        }
      }
    }

    if (eval > plim) {
      a->score = eval;
      buf[t] = a;
      cout << "prepared_player (" << restarts << " restarts): ACCEPTING " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
    } else {
      cout << "prepared_player (" << restarts << " restarts): rejecting " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
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
vector<agent_ptr> game_generator::prepare_n(agent_f gen, int n, float plim) const {
  vector<agent_ptr> buf(n);

  for (int i = 0; i < n; i++) {
    cout << "prepare_n: starting " << (i + 1) << "/" << n << endl
         << "----------------------------------------" << endl;
    agent_ptr a = 0;
    while (!a) a = prepared_player(gen, plim);
    buf[i] = a;
    cout << "prepare_n: completed " << (i + 1) << "/" << n << " score " << a->score << endl
         << "----------------------------------------" << endl;
  }

  return buf;
}
