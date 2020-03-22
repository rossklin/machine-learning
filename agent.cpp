#include <iostream>
#include <memory>
#include <sstream>

#include "agent.hpp"
#include "choice.hpp"
#include "evaluator.hpp"
#include "game.hpp"
#include "utility.hpp"

using namespace std;

int agent::idc = 0;

agent::agent() : csel(0.2) {
  id = idc++;
}

void agent::train(vector<record> results) {
  cout << "agent::train: start" << endl;
  int n = results.size();
  double gamma = 1;

  for (int i = n - 2; i >= 0; i--) {
    float r = results[i].reward;
    results[i].sum_future_rewards = r + gamma * results[i + 1].sum_future_rewards;
  }

  cout << "agent::train: update" << endl;
  int num_correct = 0, num_zero = 0, num_bad = 0;
  for (auto y : results) {
    double target = y.sum_future_rewards;
    double old_output = eval->evaluate(y.input);

    eval->update(y.input, target, age);

    double new_output = eval->evaluate(y.input);
    double change = new_output - old_output;
    double desired_change = target - old_output;

    num_correct += signum(change) == signum(desired_change);
    num_bad += signum(change) && signum(change) != signum(desired_change);
    num_zero += signum(change) == 0;
  }

  cout << "stats: " << num_correct << ", " << num_bad << ", " << num_zero << endl;
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
  return a;
}

agent_ptr agent::mutate() const {
  agent_ptr a = clone();
  a->eval->mutate();
  return a;
}

double agent::evaluate_choice(vec x) const {
  return eval->evaluate(x);
}

void agent::initialize_from_input(input_sampler s, int choice_dim) {
  eval->initialize(s, choice_dim);
};

std::string agent::serialize() const {
  // todo: serialize other agent components
  return eval->serialize();
}

void agent::deserialize(std::string s) {
  stringstream ss(s);
  eval->deserialize(ss);
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
  return eval->status_report();
}