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

template <typename E>
void agent<E>::train(vector<record> results) {
  int n = results.size();
  double gamma = 0.95;

  for (int i = n - 2; i >= 0; i--) {
    float r = results[i].reward;
    results[i].sum_future_rewards = r + gamma * results[i + 1].sum_future_rewards;
  }

  for (auto y : results) update_evaluator(y.input, y.sum_future_rewards);
}

template <typename E>
bool agent<E>::evaluator_stability() {
  return eval->stable;
}

template <typename E>
void agent<E>::set_exploration_rate(float r) {
  csel.set_exploration_rate(r);
}

template <typename E>
agent<E>::ptr agent<E>::mate(ptr p) {
  ptr a = clone();
  a->eval = p->mate_evaluator(eval);
  return a;
}

template <typename E>
agent<E>::ptr agent<E>::mutate() {
  ptr a = clone();
  a->eval->mutate();
  return a;
}

template <typename E>
double agent<E>::evaluate_choice(vec x) {
  return eval->evaluate(x);
}

template <typename E>
evaluator_ptr agent<E>::mate_evaluator(evaluator_ptr e) {
  evaluator_ptr x = eval->clone();
  x->mate(e);
  return x;
}

template <typename E>
void agent<E>::initialize_from_input(input_sampler s, int choice_dim) {
  eval->initialize(s, choice_dim);
};

template <typename E>
std::string agent<E>::serialize() {
  return eval->serialize();
}

template <typename E>
void agent<E>::deserialize(std::stringstream &s) {
  eval->deserialize(s);
}
