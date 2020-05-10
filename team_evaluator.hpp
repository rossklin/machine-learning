#pragma once

#include <vector>

#include "evaluator.hpp"
#include "types.hpp"

class team_evaluator : public evaluator {
 protected:
  std::vector<evaluator_ptr> evals;
  int role_index;

 public:
  typedef std::shared_ptr<team_evaluator> ptr;

  team_evaluator() = default;
  team_evaluator(std::vector<evaluator_ptr> e, int ri);

  double evaluate(vec x) override;
  bool update(std::vector<record> results, int age, double &rel_change) override;
  void prune(double limit = 0) override;
  evaluator_ptr mate(evaluator_ptr partner) const override;
  evaluator_ptr mutate() const override;
  std::string serialize() const override;
  void deserialize(std::stringstream &ss) override;
  void initialize(input_sampler sampler, int cdim, std::set<int> ireq) override;
  std::string status_report() const override;
  evaluator_ptr clone() const override;
  double complexity() const override;
  std::set<int> list_inputs() const override;
  void add_inputs(std::set<int> inputs) override;
  void set_learning_rate(double r) override;

  void update_stable();
};
