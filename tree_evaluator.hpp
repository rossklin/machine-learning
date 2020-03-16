#pragma once

#include <iostream>

#include "evaluator.hpp"

class tree_evaluator : public evaluator {
  enum tree_class {
    UNARY_TREE,
    BINARY_TREE,
    CONSTANT_TREE,
    INPUT_TREE
  };

  struct tree {
    typedef std::shared_ptr<tree> ptr;
    double w;
    tree_class class_id;
    double const_value;
    int input_index;
    std::string fname;
    std::vector<ptr> subtree;
    double resbuf;
    double dwbuf;
    double ssw;
    double ssdw;

    double evaluate(vec x) const;
    double get_val(vec x);
    void initialize(int dim, int depth);
    ptr get_subtree(double p_cut);
    bool emplace_subtree(ptr, double p_put);
    ptr clone();
    void update(vec x, double delta, double alpha, bool &stable);  // return SS of dw
    void apply_dw(double scale);                                   // return SS of w
    void scale_weights(double scale);                              // return SS of w
    int count_trees();
    bool descendant_exists(tree *p, int lev = 0);
    bool loop_free(int lev = 0);
    void mutate(int dim);  // mutate in place
    std::string serialize() const;
    void deserialize(std::stringstream &ss);
  };

  static hm<std::string, t_unary> unary_ops();
  static hm<std::string, t_binary> binary_ops();

 public:
  tree::ptr root;
  double learning_rate;
  double weight_limit;

  tree_evaluator();
  double evaluate(vec x) const;
  void update(vec input, double output, int age);
  evaluator_ptr mate(evaluator_ptr partner) const;
  evaluator_ptr mutate() const;
  std::string serialize() const;
  void deserialize(std::string s);
  void initialize(input_sampler sampler, int cdim);
  std::string status_report() const;
  evaluator_ptr clone() const;
  double complexity_penalty() const;
};
