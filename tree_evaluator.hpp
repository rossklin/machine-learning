#pragma once

#include "evaluator.hpp"

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
    void deserialize(std::string s);
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
  void deserialize(std::string s);
  void initialize(input_sampler sampler, int cdim);
  std::string status_report();
  ptr clone();
  double complexity_penalty();
};
