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
  enum dist_category {
    MUT_SMALL,
    MUT_MEDIUM,
    MUT_LARGE,
    MUT_RANDOM
  };

  std::string tag;
  double learning_rate;
  bool stable;
  dist_category mut_tag;
  std::vector<std::pair<double, vec>> memories;

  evaluator();
  virtual bool update(std::vector<record> records, agent_ptr a, double &rel_change);
  virtual void reset_memory_weights(double a);

  virtual double evaluate(vec x) = 0;
  virtual void prune(double limit = 0) = 0;
  virtual evaluator_ptr mate(evaluator_ptr partner) const = 0;
  virtual evaluator_ptr mutate(dist_category dc = MUT_RANDOM) const = 0;
  virtual std::string serialize() const;
  virtual void deserialize(std::stringstream &ss);
  virtual void initialize(input_sampler sampler, int cdim, std::set<int> ireq) = 0;
  virtual std::string status_report() const = 0;
  virtual evaluator_ptr clone() const = 0;
  virtual double complexity() const = 0;
  virtual std::set<int> list_inputs() const = 0;
  virtual void add_inputs(std::set<int> inputs) = 0;
  virtual void set_learning_rate(double r);

 protected:
  virtual void set_weights(const vec &x) = 0;
  virtual vec get_weights() const = 0;
  virtual vec gradient(vec input, double target, double w_reg) const = 0;
};

evaluator_ptr deserialize_evaluator(std::stringstream &ss);
std::string serialize_evaluator(evaluator_ptr e);