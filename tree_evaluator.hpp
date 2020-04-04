#pragma once

#include <memory>

#include "evaluator.hpp"

class tree_evaluator : public evaluator {
  enum tree_class {
    UNARY_TREE,     // 0
    BINARY_TREE,    // 1
    CONSTANT_TREE,  // 2
    INPUT_TREE,     // 3
    WEIGHT_TREE     // 4
  };

  struct tree : public std::enable_shared_from_this<tree> {
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

    double evaluate(const vec &x);
    void initialize(int dim, int depth);
    ptr get_subtree(double p_cut);
    void emplace_subtree(ptr, double p_put);
    ptr clone();
    void calculate_dw(double delta, double alpha, double gamma, bool &stable);  // return SS of dw
    void apply_dw(double scale);                                                // return SS of w
    void scale_weights(double scale);                                           // return SS of w
    int count_trees();
    bool descendant_exists(tree *p, int lev = 0);
    bool loop_free(int lev = 0);
    void prune();
    void mutate(int dim);  // mutate in place
    std::string serialize() const;
    void deserialize(std::stringstream &ss);
  };

  static hm<std::string, t_unary> unary_ops();
  static hm<std::string, t_binary> binary_ops();

 public:
  tree::ptr root;
  double gamma;  // regularization rate
  double learning_rate;
  double weight_limit;
  int depth;

  tree_evaluator();
  tree_evaluator(int depth);
  double evaluate(vec x);  // modifies resbuf
  bool update(vec input, double output, int age);
  evaluator_ptr mate(evaluator_ptr partner) const;
  evaluator_ptr mutate() const;
  std::string serialize() const;
  void deserialize(std::stringstream &ss);
  void initialize(input_sampler sampler, int cdim);
  void example_setup(int cdim);
  std::string status_report() const;
  evaluator_ptr clone() const;
  double complexity_penalty() const;
  double complexity() const;
};
