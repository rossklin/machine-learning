#pragma once

#include <functional>
#include <memory>
#include <set>
#include <string>

#include "types.hpp"

class evaluator {
 protected:
  int dim;

 public:
  std::string tag;
  double learning_rate;
  bool stable;

  virtual double evaluate(vec x) = 0;
  virtual bool update(std::vector<record> results, int age, double &rel_change) = 0;
  virtual void prune() = 0;
  virtual evaluator_ptr mate(evaluator_ptr partner) const = 0;
  virtual evaluator_ptr mutate() const = 0;
  virtual std::string serialize() const;
  virtual void deserialize(std::stringstream &ss);
  virtual void initialize(input_sampler sampler, int cdim, std::set<int> ireq) = 0;
  virtual std::string status_report() const = 0;
  virtual evaluator_ptr clone() const = 0;
  virtual double complexity_penalty() const = 0;
  virtual double complexity() const = 0;
  virtual std::set<int> list_inputs() const = 0;
};
