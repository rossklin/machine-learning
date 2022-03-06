#include "agent.hpp"

#include <omp.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "choice.hpp"
#include "evaluator.hpp"
#include "game.hpp"
#include "pod_agent.hpp"
#include "utility.hpp"

using namespace std;

int agent::idc = 0;

training_stats::training_stats() {
  n = 0;
  rel_change_mean = 0;
  rate_successfull = 1;
  output_change = 0;
  rate_optim_failed = 0;
}

ostream &operator<<(ostream &os, const training_stats &x) {
  return os << x.n << sep
            << x.rel_change_mean << sep
            << x.rate_successfull << sep
            << x.output_change << sep
            << x.rate_optim_failed << sep;
}

istream &operator>>(istream &is, training_stats &x) {
  return is >> x.n >> x.rel_change_mean >> x.rate_successfull >> x.output_change >> x.rate_optim_failed;
}

void auto_update(double &x, double t, double gamma, double n) {
  double hw = (1 - pow(gamma, n + 1)) / (1 - gamma) - 1;
  x = (t + hw * x) / (hw + 1);
}

string serialize_agent(agent_ptr a) {
  stringstream ss;
  ss << a->class_id << sep << a->serialize();
  return ss.str();
}

agent_ptr deserialize_agent(stringstream &ss) {
  agent_ptr a;
  int test;
  string class_tag;

  ss >> test;

  if (test == (int)POD_AGENT) {
    a = agent_ptr(new pod_agent);
    class_tag = "pod";
  } else {
    throw runtime_error("Invalid agent class id: " + to_string(test));
  }

  a->deserialize(ss);
  a->label = a->eval->tag + "-" + class_tag;

  return a;
}

agent::agent() {
  static MutexType lock;
  lock.Lock();
  id = idc++;
  lock.Unlock();

  original_id = id;
  rank = 0;
  last_rank = 0;
  was_protected = false;
  age = 0;
  mut_age = 0;
  mem_limit = fabs(rnorm(1e2, 5e1));
  mem_curve = fabs(rnorm(2e1, 1e1));
  inspiration_age_limit = fabs(rnorm(1e1, 2));
  future_discount = pow(10, -u01(1, 2));
  w_reg = pow(10, -u01(3, 5));
  learning_rate = pow(10, -u01(4, 6));
  step_limit = pow(10, -u01(2, 4));
  use_f0c = u01() < 0.1;

  csel = choice_selector_ptr(new choice_selector(0.2));
}

void agent::train(vector<vector<record>> results, input_sampler isam) {
  // Test outputs
  int ntest = 20;
  vector<vec> test_inputs(ntest);
  vector<double> test_outputs(ntest);
  for (int i = 0; i < ntest; i++) {
    record r = isam();
    test_inputs[i] = vec_append(r.opts[r.selected_option].choice, r.state);
    test_outputs[i] = eval->evaluate(test_inputs[i]);
  }

  // Calculate SFR
  double gamma = 1 - future_discount;
  for (auto &res : results) {
    int n = res.size();
    res.back().sum_future_rewards = res.back().reward;
    for (int i = n - 2; i >= 0; i--) {
      float r = res[i].reward;
      res[i].sum_future_rewards = r + gamma * res[i + 1].sum_future_rewards;
    }
  }

  // Optimize evaluator
  double rel_change = 0;
  evaluator_ptr upd = eval->update(vec_flatten(results), shared_from_this(), rel_change);
  if (upd) eval = upd;
  bool success = !!upd;

  // Test outputs
  vector<double> test_outputs2(ntest), diffs(ntest);
  for (int i = 0; i < ntest; i++) {
    test_outputs2[i] = eval->evaluate(test_inputs[i]);
    diffs[i] = test_outputs2[i] - test_outputs[i];
  }
  double output_change = l2norm(diffs) / l2norm(test_outputs);

  if (!isfinite(output_change)) {
    eval->stable = false;
  }

  age++;
  mut_age++;

  // update training stats, age and adapt learning rate
  double tsrate = 0.9;

  auto_update(tstats.rate_successfull, success, tsrate, tstats.n);

  if (success) {
    auto_update(tstats.output_change, output_change, tsrate, tstats.n);
    auto_update(tstats.rel_change_mean, rel_change, tsrate, tstats.n);

    if (tstats.output_change > 1e-2 / sqrt(age)) {
      tstats.output_change *= 0.75;
      learning_rate = 0.75 * learning_rate;
    } else if (tstats.output_change < 1e-4 / sqrt(age)) {
      tstats.output_change *= 1.25;
      learning_rate = fmin(1.25 * learning_rate, 1);
    }
  } else {
    auto_update(tstats.rate_optim_failed, !isfinite(rel_change), tsrate, tstats.n);
  }

  tstats.n++;
}

bool agent::evaluator_stability() const {
  return eval->stable;
}

void agent::set_exploration_rate(float r) {
  csel->set_exploration_rate(r);
}

double join_vals(vec x) {
  return sum(x) / x.size();
}

double mutate_val(double x) {
  return fmax(rnorm(x, 0.01 * x), 0);
}

agent_ptr agent::mate(agent_ptr p) const {
  agent_ptr a = clone();
  a->eval = p->eval->mate(eval);
  a->eval->prune();
  a->parents = {id, p->id};
  a->ancestors = set_union(ancestors, p->ancestors);
  a->ancestors.insert(id);
  a->ancestors.insert(p->id);
  a->original_id = a->id;
  a->future_discount = join_vals({future_discount, p->future_discount});
  a->w_reg = join_vals({w_reg, p->w_reg});
  a->mem_curve = join_vals({mem_curve, p->mem_curve});
  a->mem_limit = join_vals({(double)mem_limit, (double)p->mem_limit});
  a->inspiration_age_limit = join_vals({(double)inspiration_age_limit, (double)p->inspiration_age_limit});
  a->learning_rate = join_vals({(double)learning_rate, (double)p->learning_rate});
  a->step_limit = join_vals({(double)step_limit, (double)p->step_limit});
  a->use_f0c = use_f0c == p->use_f0c ? use_f0c : u01() > 0.5;
  return a->mutate();
}

agent_ptr agent::mutate() const {
  agent_ptr a = clone();
  a->eval = a->eval->mutate();
  a->eval->prune();
  a->parents = parents;
  a->ancestors = ancestors;
  a->age = age;
  a->original_id = id;

  // mutate conf params
  a->future_discount = mutate_val(a->future_discount);
  a->w_reg = mutate_val(a->w_reg);
  a->mem_curve = mutate_val(a->mem_curve);
  a->mem_limit = mutate_val(a->mem_limit);
  a->inspiration_age_limit = mutate_val(a->inspiration_age_limit);
  a->learning_rate = mutate_val(a->learning_rate);
  a->step_limit = mutate_val(a->step_limit);
  return a;
}

double agent::evaluate_choice(vec x) const {
  return eval->evaluate(x);
}

void agent::initialize_from_input(input_sampler s, int choice_dim, set<int> ireq) {
  eval->initialize(s, choice_dim, ireq);
};

std::string agent::serialize() const {
  stringstream ss;

  // classifiers
  ss << id << sep
     << class_id << sep
     << original_id << sep
     << label << sep

     // stats
     << score_tmt.serialize(sep) << sep
     << score_simple.serialize(sep) << sep
     << score_refbot.serialize(sep) << sep
     << rank << sep
     << last_rank << sep
     << age << sep
     << mut_age << sep

     // learning system parameters
     << future_discount << sep
     << w_reg << sep      // todo: regularization
     << mem_limit << sep  // todo: cap nr memories
     << mem_curve << sep  // todo: length of mem fade curve
     << inspiration_age_limit << sep

     << tstats << sep
     << parents << sep
     << ancestors << sep

     << csel->serialize() << sep
     << serialize_evaluator(eval);

  return ss.str();
}

void agent::deserialize(std::stringstream &ss) {
  ss
      // classifiers
      >> id >> class_id >> original_id >> label;

  // stats
  score_tmt.deserialize(ss);
  score_simple.deserialize(ss);
  score_refbot.deserialize(ss);

  ss >> rank >> last_rank >> age >> mut_age

      // learning system parameters
      >> future_discount >> w_reg >> mem_limit >> mem_curve >> inspiration_age_limit

      >> tstats >> parents >> ancestors;

  csel->deserialize(ss);
  eval = deserialize_evaluator(ss);

  // guarantee deserializing an agent does not break id generator
  if (id >= idc) idc = id + 1;
}

record agent::select_choice(game_ptr g) {
  record r;
  auto opts = g->generate_choices(shared_from_this());

  r.state = g->vectorize_state(id);
  r.opts.resize(opts.size());
  r.reward = 0;
  r.sum_future_rewards = 0;

  for (int i = 0; i < opts.size(); i++) {
    r.opts[i].choice = g->vectorize_choice(opts[i], id);
    r.opts[i].input = vec_append(r.opts[i].choice, r.state);
    r.opts[i].output = evaluate_choice(r.opts[i].input);
  }

  r.selected_option = csel->select(r.opts);

  return r;
}

string agent::status_report() const {
  stringstream ss;
  vector<int> parent_ids(parents.begin(), parents.end());
  string parent_hash;

  if (parents.size()) {
    parent_hash = join_string(map<int, string>([](int i) { return to_string(i); }, parent_ids), "#");
  } else {
    parent_hash = "creation";
  }

  // classifiers
  ss << id << comma
     << parent_hash << comma
     << class_id << comma
     << original_id << comma
     << label << comma

     // stats
     << score_tmt.serialize(comma) << comma
     << score_simple.serialize(comma) << comma
     << score_refbot.serialize(comma) << comma
     << rank << comma
     << last_rank << comma
     << age << comma
     << mut_age << comma

     // learning system parameters
     << future_discount << comma
     << w_reg << comma      // todo: regularization
     << mem_limit << comma  // todo: cap nr memories
     << mem_curve << comma  // todo: length of mem fade curve
     << inspiration_age_limit << comma
     << learning_rate << comma
     << step_limit << comma
     << use_f0c << comma

     // training stats
     << tstats.rel_change_mean << comma
     << tstats.output_change << comma
     << tstats.rate_successfull << comma
     << tstats.rate_optim_failed << comma

     // optimization stats
     << optim_stats.success.serialize(comma) << comma
     << optim_stats.improvement.serialize(comma) << comma
     << optim_stats.its.serialize(comma) << comma
     << optim_stats.overshoot.serialize(comma) << comma
     << optim_stats.dx.serialize(comma) << comma
     << optim_stats.dy.serialize(comma) << comma

     << parents.size() << comma
     << ancestors.size() << comma
     << eval->status_report();

  return ss.str();
}