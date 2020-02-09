#pragma once

#include <functional>
#include <memory>
#include <sstream>

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
  virtual std::string serialize();
  virtual void deserialize(std::stringstream &ss);
  virtual void initialize(input_sampler sampler, int cdim) = 0;
  virtual std::string status_report() = 0;
  virtual ptr clone() = 0;
  virtual double complexity_penalty();

  vec get_choice(const vec &input);
  vec get_state(const vec &input);
};

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

class tree_evaluator : public evaluator {
  enum tree_class {
    UNARY_TREE,
    BINARY_TREE,
    CONSTANT_TREE,
    INPUT_TREE
  };

  struct tree {
    double w;
    tree_class class_id;
    double const_value;
    int input_index;
    std::string fname;
    std::vector<std::shared_ptr<tree>> subtree;
    double resbuf;
    double dwbuf;
    double ssw;
    double ssdw;

    double evaluate(vec x);
    double get_val(vec x);
    void initialize(int dim, int depth);
    std::shared_ptr<tree>
    get_subtree(double p_cut);
    bool emplace_subtree(std::shared_ptr<tree>, double p_put);
    std::shared_ptr<tree>
    clone();
    void update(vec x, double delta, double alpha, bool &stable);  // return SS of dw
    void apply_dw(double scale);                                   // return SS of w
    void scale_weights(double scale);                              // return SS of w
    int count_trees();
    bool descendant_exists(tree *p, int lev = 0);
    bool loop_free(int lev = 0);
    void mutate(int dim);
    std::string serialize();
    void deserialize(std::stringstream &ss);
  };

  static hm<std::string, t_unary> unary_op;
  static hm<std::string, t_binary> binary_op;

 public:
  std::shared_ptr<tree> root;
  double learning_rate;
  double weight_limit;

  tree_evaluator();
  double evaluate(vec x);
  void update(vec input, double output, int age);
  ptr mate(ptr partner);
  ptr mutate();
  std::string serialize();
  void deserialize(std::stringstream &ss);
  void initialize(input_sampler sampler, int cdim);
  std::string status_report();
  ptr clone();
  double complexity_penalty();
};

typedef std::shared_ptr<rbf_evaluator> rbf_evaluator_ptr;
typedef std::shared_ptr<tree_evaluator> tree_evaluator_ptr;
