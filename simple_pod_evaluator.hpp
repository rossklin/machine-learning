#pragma once

#include "evaluator.hpp"

class simple_pod_evaluator : public evaluator {
 public:
  double evaluate(std::vector<double> x);
  void update(std::vector<double> input, double output, int age);
  evaluator_ptr mate(evaluator_ptr partner);
  evaluator_ptr mutate();
  evaluator_ptr clone();
  std::string serialize();
  void deserialize(std::string data);
  void initialize(input_sampler sampler, int cdim);
  std::string status_report();
  double complexity_penalty();
};
