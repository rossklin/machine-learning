#include "agent.hpp"

#include <omp.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>

#include "choice.hpp"
#include "evaluator.hpp"
#include "game.hpp"
#include "pod_agent.hpp"
#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

int agent::idc = 0;

training_stats::training_stats() {
  rel_change_mean = 0;
  rel_change_max = 0;
  rate_zero = 0;
  rate_accurate = 1;
  rate_correct_sign = 1;
  rate_successfull = 1;
  output_change = 0;
  rate_optim_failed = 0;
}

void auto_update(double &x, double t, double r) {
  x = r * t + (1 - r) * x;
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

  original_id = -1;
  score = 0;
  rank = 0;
  last_rank = 0;
  simple_score = 0;
  was_protected = false;
  age = 0;
  mut_age = 0;

  csel = choice_selector_ptr(new choice_selector(0.2));
}

void agent::train(vector<record> results, input_sampler isam) {
  int n = results.size();
  double gamma = 1;
  int ntest = 100;
  vector<vec> test_inputs(ntest);
  vector<double> test_outputs(ntest);
  for (int i = 0; i < ntest; i++) {
    test_inputs[i] = isam();
    test_outputs[i] = eval->evaluate(test_inputs[i]);
  }

  for (int i = n - 2; i >= 0; i--) {
    float r = results[i].reward;
    results[i].sum_future_rewards = r + gamma * results[i + 1].sum_future_rewards;
  }

  double rel_change;
  bool success = eval->update(results, age, rel_change);

  vector<double> test_outputs2(ntest), diffs(ntest);
  for (int i = 0; i < ntest; i++) {
    test_outputs2[i] = eval->evaluate(test_inputs[i]);
    diffs[i] = test_outputs2[i] - test_outputs[i];
  }
  double output_change = l2norm(diffs) / l2norm(test_outputs);

  if (!isfinite(output_change)) {
    eval->stable = false;
  }

  // update training stats
  double tsrate = 0.05;

  auto_update(tstats.rate_successfull, success, tsrate);

  if (success) {
    auto_update(tstats.rel_change_mean, rel_change, tsrate);
    auto_update(tstats.output_change, output_change, tsrate);
  } else {
    auto_update(tstats.rate_optim_failed, !isfinite(rel_change), tsrate);
  }

  age++;
  mut_age++;

  if (tstats.output_change > 1e-2 / sqrt(age)) {
    tstats.output_change *= 0.75;
    eval->learning_rate *= 0.75;
  } else if (tstats.output_change < 1e-4 / sqrt(age)) {
    tstats.output_change *= 1.25;
    eval->learning_rate *= 1.25;
    if (eval->learning_rate > 1) eval->learning_rate = 1;
  }
}

bool agent::evaluator_stability() const {
  return eval->stable;
}

void agent::set_exploration_rate(float r) {
  csel->set_exploration_rate(r);
}

agent_ptr agent::mate(agent_ptr p) const {
  agent_ptr a = clone();
  a->eval = p->eval->mate(eval);
  a->parents = {id, p->id};
  a->ancestors = set_union(ancestors, p->ancestors);
  a->ancestors.insert(id);
  a->ancestors.insert(p->id);
  a->score = 0.5 * 0.9 * (score + p->score);
  a->simple_score = 0.5 * 0.9 * (simple_score + p->simple_score);
  a->original_id = -1;
  return a;
}

agent_ptr agent::mutate() const {
  agent_ptr a = clone();
  a->eval->mutate();
  a->parents = parents;
  a->ancestors = ancestors;
  a->score = 0.9 * score;
  a->simple_score = 0.9 * simple_score;
  a->age = age;
  a->original_id = -1;
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

  ss << id << sep << score << sep << rank << sep << simple_score << sep << was_protected << sep << age << sep << mut_age << sep << ancestors << sep << parents << sep << csel->serialize() << sep << eval->tag << sep << eval->serialize();
  return ss.str();
}

void agent::deserialize(std::stringstream &ss) {
  ss >> id >> score >> rank >> simple_score >> was_protected >> age >> mut_age >> ancestors >> parents;
  csel->deserialize(ss);

  string tag;
  ss >> tag;

  if (tag == "tree") {
    eval = evaluator_ptr(new tree_evaluator);
  } else {
    throw runtime_error("Invalid evaluator tag: " + tag);
  }

  eval->deserialize(ss);

  // guarantee deserializing an agent does not break id generator
  if (id >= idc) idc = id + 1;
}

choice_ptr agent::select_choice(game_ptr g) {
  auto opts = g->generate_choices(shared_from_this());
  vec s = g->vectorize_state(id);
  for (auto opt : opts) {
    vec x = g->vectorize_choice(opt, id);
    x.insert(x.end(), s.begin(), s.end());
    opt->value_buf = evaluate_choice(x);
  }

  return csel->select(opts);
}

float agent::complexity_penalty() const {
  return eval->complexity_penalty();
}

string agent::status_report() const {
  stringstream ss;
  string comma = ",";

  ss << id << comma << label << comma << age << comma
     << score << comma << ancestors.size() << comma << parents.size() << comma
     << tstats.rel_change_mean << comma
     << tstats.output_change << comma
     << tstats.rate_successfull << comma
     << tstats.rate_optim_failed << comma
     << eval->status_report();

  return ss.str();
}