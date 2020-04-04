#include "agent.hpp"

#include <omp.h>

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

agent::agent() : csel(0.2) {
  static MutexType lock;
  lock.Lock();
  id = idc++;
  lock.Unlock();

  original_id = -1;
  score = 0;
  last_score = 0;
  simple_score = 0;
  was_protected = false;
  age = 0;
  mut_age = 0;
}

void agent::train(vector<record> results) {
  int n = results.size();
  double gamma = 1;

  for (int i = n - 2; i >= 0; i--) {
    float r = results[i].reward;
    results[i].sum_future_rewards = r + gamma * results[i + 1].sum_future_rewards;
  }

  int num_correct = 0, num_zero = 0, num_bad = 0, num_success = 0;
  for (auto y : results) {
    double target = y.sum_future_rewards;
    double old_output = eval->evaluate(y.input);

    num_success += eval->update(y.input, target, age);

    double new_output = eval->evaluate(y.input);
    double change = new_output - old_output;
    double desired_change = target - old_output;

    num_correct += signum(change) == signum(desired_change);
    num_bad += signum(change) && signum(change) != signum(desired_change);
    num_zero += signum(change) == 0;
  }

  if (num_success == 0) eval->stable = false;  // evaluator can no longer update
}

bool agent::evaluator_stability() const {
  return eval->stable;
}

void agent::set_exploration_rate(float r) {
  csel.set_exploration_rate(r);
}

agent_ptr agent::mate(agent_ptr p) const {
  agent_ptr a = clone();
  a->eval = p->eval->mate(eval);
  a->parents = {id, p->id};
  a->ancestors = set_union(ancestors, p->ancestors);
  a->ancestors.insert(id);
  a->ancestors.insert(p->id);
  a->score = a->last_score = 0.5 * 0.9 * (score + p->score);
  a->simple_score = 0.5 * 0.9 * (simple_score + p->simple_score);
  a->original_id = -1;
  return a;
}

agent_ptr agent::mutate() const {
  agent_ptr a = clone();
  a->eval->mutate();
  a->parents = parents;
  a->ancestors = ancestors;
  a->score = a->last_score = 0.9 * score;
  a->simple_score = 0.9 * simple_score;
  a->age = age;
  a->original_id = -1;
  return a;
}

double agent::evaluate_choice(vec x) const {
  return eval->evaluate(x);
}

void agent::initialize_from_input(input_sampler s, int choice_dim) {
  eval->initialize(s, choice_dim);
};

std::string agent::serialize() const {
  stringstream ss;

  ss << id << sep << score << sep << last_score << sep << simple_score << sep << was_protected << sep << age << sep << mut_age << sep << ancestors << sep << parents << sep << csel.serialize() << sep << eval->tag << sep << eval->serialize();
  return ss.str();
}

void agent::deserialize(std::stringstream &ss) {
  ss >> id >> score >> last_score >> simple_score >> was_protected >> age >> mut_age >> ancestors >> parents;
  csel.deserialize(ss);

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
  for (auto opt : opts) opt->value_buf = evaluate_choice(g->vectorize_choice(opt, id));
  return csel.select(opts);
}

float agent::complexity_penalty() const {
  return eval->complexity_penalty();
}

string agent::status_report() const {
  stringstream ss;
  string comma = ",";

  ss << id << comma << label << comma << age << comma << score << comma << ancestors.size() << comma << parents.size() << comma << eval->status_report();

  return ss.str();
}