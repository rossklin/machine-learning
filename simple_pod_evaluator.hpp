#pragma once

#include "evaluator.hpp"
#include "types.hpp"

class simple_pod_evaluator : public evaluator {
 public:
  double evaluate(vec x) override;
  bool update(std::vector<record> results, int age, double &rel_change) override;
  void prune() override;
  evaluator_ptr mate(evaluator_ptr partner) const override;
  evaluator_ptr mutate() const override;
  evaluator_ptr clone() const override;
  std::string serialize() const override;
  void deserialize(std::stringstream &ss) override;
  void initialize(input_sampler sampler, int cdim, std::set<int> ireq) override;
  std::string status_report() const override;
  double complexity_penalty() const override;
  double complexity() const;
  std::set<int> list_inputs() const;
};
