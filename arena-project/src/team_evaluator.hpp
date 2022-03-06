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

  team_evaluator();
  team_evaluator(std::vector<evaluator_ptr> e, int ri);

  double evaluate(vec x) override;
  evaluator_ptr update(std::vector<record> records, agent_ptr a, double &rel_change) const override;
  void reset_memory_weights(double a) override;
  void prune(double limit = 0) override;
  evaluator_ptr mate(evaluator_ptr partner) const override;
  evaluator_ptr mutate(dist_category dc) const override;
  std::string serialize() const override;
  void deserialize(std::stringstream &ss) override;
  void initialize(input_sampler sampler, int cdim, std::set<int> ireq) override;
  std::string status_report() const override;
  evaluator_ptr clone() const override;
  double complexity() const override;
  std::set<int> list_inputs() const override;
  void add_inputs(std::set<int> inputs) override;
  void set_weights(const vec &x) override;
  vec get_weights() const override;
  vec gradient(vec input, double delta) const override;

  void update_stable();
};
