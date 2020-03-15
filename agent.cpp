#include <memory>

#include "agent.hpp"
#include "choice.hpp"
#include "evaluator.hpp"
#include "game.hpp"

using namespace std;

int agent::idc = 0;

agent::agent() : csel(0.2) {
  id = idc++;
}

void agent::train(vector<record> results) {
  int n = results.size();
  double gamma = 0.95;

  for (int i = n - 2; i >= 0; i--) {
    float r = results[i].reward;
    results[i].sum_future_rewards = r + gamma * results[i + 1].sum_future_rewards;
  }

  for (auto y : results) eval->update(y.input, y.sum_future_rewards, age);
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
  return eval->serialize();
}

void agent::deserialize(std::string s) {
  eval->deserialize(s);
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