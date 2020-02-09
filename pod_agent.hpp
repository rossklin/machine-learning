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

class pod_agent : public standard_agent {
 public:
  pod_data data;

  pod_agent(evaluator_ptr e, choice_selector_ptr c);
  agent_ptr clone() override;
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

typedef std::shared_ptr<pod_agent> pod_agent_ptr;
