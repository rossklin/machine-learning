#include "pod_game.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>

#include "agent.hpp"
#include "evaluator.hpp"
#include "pod_agent.hpp"
#include "utility.hpp"

using namespace std;
using namespace pod_game_parameters;

void pod_game::initialize() {
  reset();

  run_laps = rand_int(1, 3);

  // generate checkpoints
  int n = rand_int(2, 5);
  checkpoint.resize(n);
  for (int i = 0; i < n; i++) checkpoint[i] = {width * u01(), height * u01()};

  // generate pods
  double a01 = point_angle(checkpoint[1] - checkpoint[0]);
  point v0 = normv(a01 + M_PI / 2);
  int pc = 0;

  auto gen_pod = [this, &pc, v0, a01]() -> pod_data {
    point start = checkpoint[0] + (players.size() / (double)2 - pc - 0.5) * 2 * pod_radius * v0;
    pc++;

    // point x;
    // point v;
    // double a;
    // int passed_checkpoint;
    // int previous_checkpoint
    // int lap;
    // int boost_count;
    // int shield_active;

    return {start, {0, 0}, a01, 0, 0, 0, 1, 0};
  };

  for (auto x : typed_agents) x.second->data = gen_pod();
}

void pod_game::setup_from_input(istream &s) {
  int ncp;

  s >> run_laps >> ncp;
  assert(ncp > 1 && ncp < 9);
  checkpoint.resize(ncp);

  for (auto &p : checkpoint) s >> p.x >> p.y;
}

int pod_choice::vector_dim() { return 4; }
bool pod_choice::validate() { return true; }

pod_game::pod_game(player_table pl) : game(pl) {
  max_turns = 300;
  for (auto x : players) {
    typed_agents[x.first] = static_pointer_cast<pod_agent>(x.second);
  }
}

record_table pod_game::increment(string row_prefix) {
  record_table res;
  hm<int, double> htab_before = htable();
  auto agents = typed_agents;

  auto process_pod = [this](shared_ptr<pod_choice> x, pod_data *p) {
  };

  // proceess choices
  for (auto x : agents) {
    int pid = x.first;
    pod_agent::ptr p = x.second;

    shared_ptr<pod_choice> c = static_pointer_cast<pod_choice>(p->select_choice(shared_from_this()));
    res[pid].input = vectorize_input(c, pid);
    res[pid].output = p->evaluate_choice(res[x.first].input);

    p->data.a += fmin(angular_speed, fabs(c->angle)) * signum(c->angle);

    if (p->data.shield_active) {
      p->data.shield_active--;
    } else if (c->shield) {
      p->data.shield_active = 3;
    } else {
      if (c->boost) {
        p->data.boost_count = false;
        c->thrust = 650;
      }
      p->data.v = p->data.v + c->thrust * normv(p->data.a);
    }

    p->data.x = p->data.x + p->data.v;
    // todo: collision here?
    p->data.v = friction * p->data.v;
    p->data.v = truncate_point(p->data.v);
    p->data.x = truncate_point(p->data.x);
  }

  // check collisions
  vector<pod_data *> check;
  for (auto &x : agents) check.push_back(&x.second->data);

  int n = check.size();
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (distance(check[i]->x, check[j]->x) < 2 * pod_radius) {
        double m1 = pod_mass * (check[i]->shield_active ? 10 : 1);
        double m2 = pod_mass * (check[j]->shield_active ? 10 : 1);
        point v1 = check[i]->v, v2 = check[j]->v;
        point x1 = check[i]->x, x2 = check[j]->x;

        double proj = sproject(v1 - v2, x1 - x2);

        if (!(isfinite(proj) && fabs(proj) < 1e6)) {
          // pods standing on top of each other
          check[i]->x = check[i]->x + point{rnorm(0, 10), rnorm(0, 10)};
          check[j]->x = check[j]->x + point{rnorm(0, 10), rnorm(0, 10)};
          continue;
        }

        // check that they are in fact travelling towards each other
        if (proj >= 0) continue;

        // elastic collision formula from wikipedia
        // todo: apply condition: minimum impulse: 120 (whatever that means)
        check[i]->v = v1 - 2 * m2 / (m1 + m2) * proj * (x1 - x2);
        check[j]->v = v2 - 2 * m1 / (m1 + m2) * proj * (x2 - x1);
      }
    }
  }

  // update checkpoint and lap info
  for (auto x : agents) {
    pod_agent::ptr p = x.second;
    int pid = x.first;

    int idx = modulo(p->data.passed_checkpoint + 1, (int)checkpoint.size());
    if (distance(p->data.x, checkpoint[idx]) < checkpoint_radius) {
      p->data.passed_checkpoint = idx;
      if (idx == 0) {
        p->data.lap++;

        if (p->data.lap >= run_laps) {
          winner = p->team;
          did_finish = true;
        }
      }
    }
  }

  hm<int, double> htab_after = htable();

  auto score = [this](int pid, hm<int, double> tab) {
    double own = tab[pid];
    double res = INFINITY;
    for (auto x : tab) {
      if (players[x.first]->team != players.at(pid)->team) res = fmin(res, own - x.second);
    }

    assert(isfinite(res));

    return res;
  };

  for (auto x : players) {
    // change in greatest distance travelled in team
    res[x.first].reward_simple = 1e-3 * (htab_after[x.first] - htab_before[x.first]);

    // change in difference in distance travelled between your team and best opponent team
    res[x.first].reward = 1e-3 * (score(x.first, htab_after) - score(x.first, htab_before));

    res[x.first].sum_future_rewards = 0;
  }

  // todo: validate that pods pass checkpoints and laps
  if (enable_output) {
    string cp_xs = join_string(map<point, string>([](point a) -> string { return to_string(int(a.x)); }, checkpoint), " ");
    string cp_ys = join_string(map<point, string>([](point a) -> string { return to_string(int(a.y)); }, checkpoint), " ");

    // write csv output
    for (auto x : typed_agents) {
      int pid = x.first;
      pod_agent::ptr p = x.second;
      (*enable_output) << row_prefix
        << game_id << comma
        << turns_played << comma
        << p->team << comma
        << p->data.lap << comma
        << pid << comma
        << p->data.x.x << comma
        << p->data.x.y << comma
        << p->data.a << comma
        << p->data.shield_active << comma
        << p->data.boost_count << comma
        << res[pid].reward << comma
        << cp_xs << comma
        << cp_ys << endl;
    }
  }

  return res;
}

bool pod_game::finished() {
  return did_finish;
}

std::string pod_game::end_stats() {
  stringstream ss;

  auto h = htable();
  int pid = team_clone_ids(0).front();
  int pid2 = team_clone_ids(1).front();

  ss << finished() << comma << (h[pid] / h[pid2]) << comma << (h[pid] / turns_played);

  return ss.str();
}

int pod_game::select_winner() {
  if (winner > -1) {
    return winner;
  } else {
    hm<int, double> ttab = ttable();
    vector<int> keys = hm_keys<int, double>(ttab);
    return winner = best_key<int, double>(keys, [ttab](int t) { return ttab.at(t); });
  }
}

double pod_game::winner_reward(int epoch) {
  return 3;
}

// simple score should be somewhere around 1
double pod_game::score_simple(int pid) {
  double standard_speed = 100;
  return pod_distance_travelled(pid) / turns_played / standard_speed;
}

void pod_game::reset() {
  game::reset();
  did_finish = false;
  run_laps = 0;
}

// agent_ptr pod_game::generate_player() {
//   if (u01() < 0.5) {
//     return tree_pod_agent();
//   } else {
//     return rbf_pod_agent();
//   }
// }

// agent_ptr pod_game::generate_refbot() {
//   return simple_pod_agent(evaluator_ptr(new simple_pod_evaluator));
// }

std::vector<choice_ptr> pod_game::generate_choices(agent_ptr p_base) {
  pod_agent::ptr p = static_pointer_cast<pod_agent>(p_base);
  vector<pod_choice> opts;
  double thrust_limit = 100 * (1 - p->complexity_penalty());

  pod_choice cx;
  for (double a = -angular_speed; a <= angular_speed; a += angular_speed / 3) {
    cx.angle = a;
    cx.boost = false;
    cx.shield = false;
    for (double t = 0; t <= 100; t += 20) {
      cx.thrust = fmin(t, thrust_limit);
      opts.push_back(cx);
    }
    cx.boost = true;
    if (p->data.boost_count > 0) opts.push_back(cx);
  }

  cx.angle = 0;
  cx.boost = false;
  cx.thrust = 0;
  cx.shield = true;
  opts.push_back(cx);

  vector<choice_ptr> res(opts.size());
  for (int i = 0; i < opts.size(); i++) res[i] = choice_ptr(new pod_choice(opts[i]));
  return res;
}

vec pod_game::vectorize_state(int pid) const {
  assert(players.count(pid) > 0);

  // relative pod data: 13 datapoints
  pod_data a = typed_agents.at(pid)->data;
  vector<double> x(typed_agents.size() * 13 + 1);
  int idx = 0;

  // add the team index
  x[idx++] = typed_agents.at(pid)->team_index;  // 4

  auto add_pod = [this, a, &x, &idx](pod_data b) {
    auto relad = [&](point p) {
      x[idx++] = angle_difference(point_angle(p - a.x), a.a);
      x[idx++] = distance(a.x, p);
    };

    relad(b.x);                                          // 5-6
    relad(a.x + b.v);                                    // 7-8
    relad(get_checkpoint(b.passed_checkpoint + 1));      // 9-10
    relad(get_checkpoint(b.passed_checkpoint + 2));      // 11-12
    x[idx++] = run_laps - b.lap;                         // 13
    x[idx++] = checkpoint.size() - b.passed_checkpoint;  // 14
    x[idx++] = angle_difference(b.a, a.a);               // 15
    x[idx++] = b.shield_active;                          // 16
    x[idx++] = b.boost_count;                            // 17
  };

  // add self
  add_pod(a);

  // add team members ordered by team index
  vector<pod_agent::ptr> buf;
  for (auto y : typed_agents) {
    if (y.first != pid && y.second->team == players.at(pid)->team) {
      buf.push_back(y.second);
    }
  }
  if (buf.size() > 1) sort(buf.begin(), buf.end(), [](agent_ptr a, agent_ptr b) { return a->team_index < b->team_index; });
  for (auto b : buf) add_pod(b->data);  // 18-30

  // add opponents
  for (auto y : typed_agents) {
    if (y.first != pid && y.second->team != players.at(pid)->team) {
      add_pod(y.second->data);  // 31-56
    }
  }

  return x;
}

vec pod_game::vectorize_choice(choice_ptr c_base, int pid) const {
  auto c = static_pointer_cast<pod_choice>(c_base);
  return {c->angle, c->thrust, (double)c->boost, (double)c->shield};
}

// protected members

double pod_game::pod_distance_travelled(int pid) {
  pod_data p = typed_agents.at(pid)->data;
  hm<int, double> dtab;

  double dsum = 0;
  int ncheck = checkpoint.size();
  for (int i = 0; i < ncheck; i++) {
    dtab[i] = distance(get_checkpoint(i), get_checkpoint(i + 1));
    dsum += dtab[i];
  }

  double travel = dsum * p.lap;
  for (int i = 0; i <= p.passed_checkpoint; i++) travel += dtab.at(i);
  travel -= distance(p.x, get_checkpoint(p.passed_checkpoint + 1));
  assert(isfinite(travel));

  return travel;
}

// table of heuristic score for all teams
hm<int, double> pod_game::ttable() {
  hm<int, double> ttab;

  for (auto p : players) ttab[p.second->team] = -INFINITY;
  for (auto p : typed_agents) ttab[p.second->team] = fmax(ttab[p.second->team], pod_distance_travelled(p.first));

  return ttab;
}

// table of heuristic score for all players
hm<int, double> pod_game::htable() {
  hm<int, double> ttab, htab;
  ttab = ttable();
  for (auto p : players) htab[p.first] = ttab[p.second->team];
  return htab;
}

point pod_game::get_checkpoint(int idx) const { return checkpoint.at(modulo(idx, (int)checkpoint.size())); }
