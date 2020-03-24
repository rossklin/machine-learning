#include <cassert>
#include <set>

#include "agent.hpp"
#include "game.hpp"
#include "game_generator.hpp"
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
  vec input = g->vectorize_choice(c, p->id);
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

#pragma omp parallel for
  for (int t = 0; t < prep_npar; t++) {
    float eval = 0;
    agent_ptr a = gen();

    for (int i = 0; i < 100; i++) {
      a->set_exploration_rate(0.4 - 0.3 * i / (float)100);

      // play and train
      game_ptr g = team_bots_vs(a);
      const auto res = g->play(i + 1);

      for (auto pid : g->team_pids(a->team)) a->train(res.at(pid));  // look at own mistakes

      if (a->eval->complexity_penalty() < 1e-5) {
        // this agent has degenerated
        a = gen();
        eval = 0;
        continue;
      }

      if (!a->eval->stable) {
        // this agent has degenerated
        a = gen();
        eval = 0;
        continue;
      }

      for (auto clone_pid : g->team_pids(a->team)) {
        eval = 0.1 * g->score_simple(clone_pid) + 0.9 * eval;
      }
    }

    if (eval > plim) {
      a->score = eval;
      buf[t] = a;
      cout << "prepared_player: ACCEPTING " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
    } else {
      cout << "prepared_player: rejecting " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
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
