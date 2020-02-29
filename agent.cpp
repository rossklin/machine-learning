#include <memory>

#include "agent.hpp"
#include "choice.hpp"
#include "evaluator.hpp"
#include "game.hpp"

using namespace std;

template <typename E>
agent<E>::agent() {
  static int idc = 0;
  id = idc++;
  eval = new E;
}

void agent::train(vector<record> results) {
  int n = results.size();
  double gamma = 0.95;

  for (int i = n - 2; i >= 0; i--) {
    float r = results[i].reward;
    results[i].sum_future_rewards = r + gamma * results[i + 1].sum_future_rewards;
  }

  for (auto y : results) update_evaluator(y.input, y.sum_future_rewards);
}

bool agent::evaluator_stability() {
  return eval->stable;
}

choice_ptr agent::select_choice(game_ptr g) {
  auto opts = g->generate_choices(shared_from_this());
  for (auto opt : opts) opt->value_buf = eval->evaluate(g->vectorize_choice(opt, id));
  return csel.select(opts);
}

void agent::set_exploration_rate(float r) {
  csel.set_exploration_rate(r);
}

agent_ptr agent::mate(agent_ptr p) {
  agent_ptr a = clone();
  a->eval = p->mate_evaluator(eval);
  return a;
}

agent_ptr agent::mutate() {
  agent_ptr a = clone();
  a->eval->mutate();
  return a;
}

double agent::evaluate_choice(vec x) {
  return eval->evaluate(x);
}

evaluator_ptr agent::mate_evaluator(evaluator_ptr e) {
  evaluator_ptr x = eval->clone();
  x->mate(e);
  return x;
}

void agent::initialize_from_input(input_sampler s, int choice_dim) {
  eval->initialize(s, choice_dim);
};

std::string agent::serialize() {
  return eval->serialize();
}

void agent::deserialize(std::stringstream &s) {
  eval->deserialize(s);
}
