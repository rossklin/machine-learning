#pragma once

#include "evaluator.hpp"
#include "types.hpp"

class simple_pod_evaluator : public evaluator {
 public:
  simple_pod_evaluator();
  double evaluate(vec x) override;
  bool update(std::vector<record> records, agent_ptr a, double &rel_change) override;
  void prune(double limit = 0) override;
  evaluator_ptr mate(evaluator_ptr partner) const override;
  evaluator_ptr mutate(dist_category dc) const override;
  evaluator_ptr clone() const override;
  std::string serialize() const override;
  void deserialize(std::stringstream &ss) override;
  void initialize(input_sampler sampler, int cdim, std::set<int> ireq) override;
  std::string status_report() const override;
  double complexity() const override;
  std::set<int> list_inputs() const override;
  void add_inputs(std::set<int> inputs) override;
  void set_learning_rate(double r) override;
  void set_weights(const vec &x) override;
  vec get_weights() const override;
  vec gradient(vec input, double target, double w_reg) const override;
};
