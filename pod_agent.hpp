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

class pod_agent : public agent {
 public:
  typedef std::shared_ptr<pod_agent> ptr;
  pod_data data;

  pod_agent() = default;
  agent_ptr clone() const override;
};
