#include <cassert>

#include "agent.hpp"
#include "game.hpp"
#include "game_generator.hpp"
#include "utility.hpp"

using namespace std;

game_generator::game_generator(int teams, int ppt, agent_f refbot_generator) : nr_of_teams(teams), ppt(ppt), refbot_generator(refbot_generator) {}

vector<agent_ptr> game_generator::make_teams(vector<agent_ptr> ps) const {
  vector<agent_ptr> buf(nr_of_teams * ppt);

  for (int filter_tid = 0; filter_tid < ps.size(); filter_tid++) {
    agent_ptr b = ps[filter_tid];
    b->team = filter_tid;
    for (int k = 0; k < ppt; k++) {
      agent_ptr a = b->clone();
      a->team = filter_tid;
      a->team_index = k;
      buf[filter_tid * ppt + k] = a;
    }
  }

  return buf;
}

game_ptr game_generator::team_bots_vs(agent_ptr a) const {
  vector<agent_ptr> ps(nr_of_teams);
  ps[0] = a;
  for (int i = 1; i < nr_of_teams; i++) ps[i] = refbot_generator();
  ps = make_teams(ps);
  return generate_starting_state(ps);
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
  int npar = 8;
  vector<agent_ptr> buf(npar);

#pragma omp parallel for
  for (int t = 0; t < npar; t++) {
    float eval = 0;
    agent_ptr a = gen();

    for (int i = 0; i < 100; i++) {
      a->set_exploration_rate(0.5 - 0.4 * i / (float)100);

      // play and train
      game_ptr g = team_bots_vs(a);
      const auto res = g->play(1);
      a->train(res.at(a->id));
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
vector<agent_ptr> game_generator::prepare_n(agent_f gen, int n, float plim) const {
  vector<agent_ptr> buf(n);

  for (int i = 0; i < n; i++) {
    cout << "prepare_n: starting " << (i + 1) << "/" << n << endl;
    agent_ptr a = 0;
    while (!a) a = prepared_player(gen, plim);
    buf[i] = a;
    cout << "prepare_n: completed " << (i + 1) << "/" << n << endl;
  }

  return buf;
}
