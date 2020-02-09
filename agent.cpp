#include <memory>

#include "agent.hpp"
#include "choice.hpp"
#include "evaluator.hpp"

using namespace std;

agent::agent() {
  static int idc = 0;
  id = idc++;
}

standard_agent::standard_agent(evaluator_ptr e, choice_selector_ptr c) : eval(e), csel(c), agent() {}

standard_agent_ptr standard_agent::clone_std() {
  return static_pointer_cast<standard_agent>(clone());
}

bool standard_agent::evaluator_stability() {
  return eval->stable;
}

choice_ptr standard_agent::select_choice(game_ptr g) {
  return csel->select(g, shared_from_this());
}

void standard_agent::set_exploration_rate(float r) {
  csel->set_exploration_rate(r);
}

agent_ptr standard_agent::mate(agent_ptr p) {
  standard_agent_ptr a = clone_std();
  a->eval = p->mate_evaluator(eval);
  return a;
}

agent_ptr standard_agent::mutate() {
  standard_agent_ptr a = clone_std();
  a->eval->mutate();
  return a;
}

double standard_agent::evaluate_choice(vec x) {
  return eval->evaluate(x);
}

evaluator_ptr standard_agent::mate_evaluator(evaluator_ptr e) {
  evaluator_ptr x = eval->clone();
  x->mate(e);
  return x;
}

void standard_agent::initialize_from_input(input_sampler s, int choice_dim) {
  eval->initialize(s, choice_dim);
};

std::string standard_agent::serialize() {
  return eval->serialize();
}
void standard_agent::deserialize(std::stringstream &s) {
  eval->deserialize(s);
}
