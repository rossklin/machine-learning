#pragma once

#include <functional>
#include <memory>
#include <string>

#include "types.hpp"

class evaluator {
 protected:
  int dim;
  int choice_dim;

 public:
  typedef evaluator_ptr ptr;
  bool stable;

  virtual double evaluate(vec x) = 0;
  virtual void update(vec input, double output, int age) = 0;
  virtual ptr mate(ptr partner) = 0;
  virtual ptr mutate() = 0;
  virtual std::string serialize() = 0;
  virtual void deserialize(std::string ss) = 0;
  virtual void initialize(input_sampler sampler, int cdim) = 0;
  virtual std::string status_report() = 0;
  virtual ptr clone() = 0;
  virtual double complexity_penalty() = 0;
};
