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
  bool stable;

  virtual double evaluate(vec x) const = 0;
  virtual void update(vec input, double output, int age) = 0;
  virtual evaluator_ptr mate(evaluator_ptr partner) const = 0;
  virtual evaluator_ptr mutate() const = 0;
  virtual std::string serialize() const = 0;
  virtual void deserialize(std::string ss) = 0;
  virtual void initialize(input_sampler sampler, int cdim) = 0;
  virtual std::string status_report() const = 0;
  virtual evaluator_ptr clone() const = 0;
  virtual double complexity_penalty() const = 0;
};
