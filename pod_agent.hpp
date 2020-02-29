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
  typedef std::shared_ptr<pod_agent<E>> ptr;
  pod_data data;

  pod_agent();
  typename agent<E>::ptr clone() override;
};
