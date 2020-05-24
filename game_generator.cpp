
#include "game_generator.hpp"

#include <algorithm>
#include <cassert>
#include <future>
#include <set>
#include <thread>

#include "agent.hpp"
#include "game.hpp"
#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

game_generator::game_generator(int teams, int ppt, agent_f refbot_generator) : nr_of_teams(teams), ppt(ppt), refbot_generator(refbot_generator) {
  prep_npar = 6;
  max_turns = 100;
  max_complexity = 800;
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
agent_ptr game_generator::prepared_player(input_sampler isam, agent_f gen, float plim) const {
  int nt = min(prep_npar, max((int)thread::hardware_concurrency() - 1, 1));

  auto task = [this, gen, plim, isam]() -> agent_ptr {
    auto vgen = [this, gen]() -> agent_ptr {
      agent_ptr a = gen();

      set<int> missing = set_difference(required_inputs(), a->eval->list_inputs());
      a->eval->add_inputs(missing);

      double wlim = 0;
      while (a->eval->complexity() > max_complexity) a->eval->prune(wlim += 1e-3);

      return a;
    };

    // #pragma omp parallel for
    //   for (int t = 0; t < prep_npar; t++) {
    float eval = 0;
    agent_ptr a = vgen();
    int restarts = 0;
    int max_its = 10;
    int ndata = 0;

    for (int i = 0; i < max_its; i++) {
      bool supervizion = i % 2 == 0 && i < 6;
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
      const auto recs = hm_values(g->play(i + 1));

      a->train(recs, isam);

      bool complex = a->eval->complexity() > 20;
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
        for (auto clone_pid : hm_keys(g->players)) {
          double res = g->score_simple(clone_pid);
          ndata += i;
          eval = (res * i + (ndata - i) * eval) / ndata;
        }
      }
    }

    if (eval > plim) {
      a->score_simple.push(eval);
      cout << "prepared_player (" << restarts << " restarts): ACCEPTING " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
      return a;
    } else {
      cout << "prepared_player (" << restarts << " restarts): rejecting " << a->id << " with complexity " << a->eval->complexity() << " at eval = " << eval << endl;
      return NULL;
    }
  };

  vector<shared_future<agent_ptr>> futs, futs_buf;
  vector<agent_ptr> buf;
  bool done = false;

  do {
    futs_buf.clear();
    for (auto f : futs) {
      if (f.wait_for(chrono::milliseconds(0)) == future_status::ready) {
        agent_ptr test = f.get();
        if (test) {
          done = true;
          buf.push_back(test);
        }
      } else {
        futs_buf.push_back(f);
      }
    }
    futs = futs_buf;

    this_thread::sleep_for(chrono::milliseconds(10));

    if (!done) {
      while (futs.size() < nt) futs.push_back(async(launch::async, task));
    }
  } while (futs.size());

  sort(buf.begin(), buf.end(), [](agent_ptr a, agent_ptr b) -> bool {
    double sa = a ? a->score_simple.current : 0;
    double sb = b ? b->score_simple.current : 0;
    return sa > sb;
  });

  return buf[0];
};

// Repeatedly attempt to make prepared players until n have been generated
vector<agent_ptr> game_generator::prepare_n(agent_f gen, int n, float plim) const {
  vector<agent_ptr> buf(n);
  input_sampler isam = generate_input_sampler();

  for (int i = 0; i < n; i++) {
    cout << "prepare_n: starting " << (i + 1) << "/" << n << endl
         << "----------------------------------------" << endl;
    buf[i] = prepared_player(isam, gen, plim);
    cout << "prepare_n: completed " << (i + 1) << "/" << n << " score " << buf[i]->score_simple.current << endl
         << "----------------------------------------" << endl;
  }

  return buf;
}
