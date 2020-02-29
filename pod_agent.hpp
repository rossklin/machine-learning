#pragma once

#include <string>
#include <vector>

#include "agent.hpp"
#include "evaluator.hpp"
#include "types.hpp"

// AGENT

struct pod_data {
  point x;
  point v;
  double a;
  int passed_checkpoint;
  int previous_checkpoint;
  int lap;
  int boost_count;
  int shield_active;
};

template <typename E>
class pod_agent : public agent<E> {
 public:
  typedef std::shared_ptr<pod_agent> ptr;
  pod_data data;

  pod_agent();
  ptr clone() override;
};

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
};
