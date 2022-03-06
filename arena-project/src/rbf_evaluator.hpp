#pragma once

#include "evaluator.hpp"

class rbf_evaluator : public evaluator {
  struct rbf_base {
    vec x_choice;
    vec x_state;
    double q;
  };

  struct rbf_stat {
    vec dc;
    vec ds;
    vec wc;
    vec ws;
    rbf_stat(int n);
  };

  std::vector<rbf_base> bases;
  vec choice_mean, choice_std, state_mean, state_std;
  double dq_mean;
  double dq_std;
  double bw_state;
  double bw_choice;
  double cluster_rate;
  double alpha;

 public:
  double evaluate(vec x);
  void update(vec input, double output, int age);
  ptr mate(ptr partner);
  ptr mutate();
  std::string serialize();
  void deserialize(std::stringstream &ss);
  void initialize(input_sampler sampler, int cdim);
  std::string status_report();
  ptr clone();

  vec normchoice(vec x);
  vec normstate(vec x);
  rbf_stat base_stats(vec xc, vec xs);
};
