#include <cassert>
#include <functional>
#include <vector>

#include "agent.hpp"
#include "game.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

game::game(int nr_of_teams, int ppt) : nr_of_teams(nr_of_teams), ppt(ppt) {
  assert(nr_of_teams > 0);
  assert(ppt > 0);
}

vector<agent_ptr> game::make_teams(vector<agent_ptr> ps) {
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

game_ptr game::team_bots_vs(agent_ptr a) {
  vector<agent_ptr> ps(nr_of_teams);
  ps[0] = a;
  for (int i = 1; i < nr_of_teams; i++) ps[i] = generate_refbot();
  ps = make_teams(ps);
  return generate_starting_state(ps);
}

int game::choice_dim() {
  game_ptr g = team_bots_vs(generate_refbot());
  agent_ptr p = players.begin()->second;
  vector<choice_ptr> cs = generate_choices(p);
  vec input = vectorize_choice(cs.front(), p->id);
  return input.size();
}

function<vec()> game::generate_input_sampler() {
  game_ptr g = team_bots_vs(generate_refbot());
  play();
  auto buf = results;

  return [buf]() -> vec {
    return buf[rand_int(0, buf.size() - 1)].begin()->second.input;
  };
}

void game::train(int pid, int filter_pid, int filter_tid) {
  int n = results.size();
  double gamma = 0.95;
  agent_ptr a = players.at(pid);

  // add reward for winning
  for (auto &x : results[n - 1]) {
    x.second.sum_future_rewards = (players[x.first]->team == winner) * reward_win();
  }

  for (int i = n - 2; i >= 0; i--) {
    for (auto &x : results[i]) {
      float r = x.second.reward;
      x.second.sum_future_rewards = r + gamma * results[i + 1][x.first].sum_future_rewards;
    }
  }

  // local training
  bool did_train = false;
  for (auto x : players) {
    if (filter_tid > -1 && x.second->team != filter_tid) continue;
    if (filter_pid > -1 && x.first != filter_pid) continue;

    did_train = true;
    for (int i = 0; i < n; i++) {
      record y = results[i][x.first];
      a->update_evaluator(y.input, y.sum_future_rewards);
    }
  }

  assert(did_train);
}