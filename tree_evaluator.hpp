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

    double evaluate(const vec &x);
    void initialize(std::vector<int> inputs);
    ptr get_subtree(double p_cut);
    void emplace_subtree(ptr, double p_put);
    ptr clone();
    int calculate_dw(vec &dgdw, int offset, double delta, double alpha, double gamma);
    int set_weights(const vec &x, int offset = 0);
    vec get_weights() const;
    int count_trees();
    bool descendant_exists(tree *p, int lev = 0);
    bool loop_free(int lev = 0);
    void prune();
    void mutate(int dim);  // mutate in place
    std::string serialize() const;
    void deserialize(std::stringstream &ss);
    std::set<int> list_inputs() const;
  };

  static const hm<std::string, t_unary> &unary_ops();
  static const hm<std::string, t_binary> &binary_ops();

 public:
  tree::ptr root;
  double gamma;  // regularization rate
  double weight_limit;
  int depth;

  tree_evaluator();
  tree_evaluator(int depth);
  double evaluate(vec x) override;  // modifies resbuf
  bool update(std::vector<record> results, int age, double &rel_change) override;
  void prune() override;
  evaluator_ptr mate(evaluator_ptr partner) const override;
  evaluator_ptr mutate() const override;
  std::string serialize() const override;
  void deserialize(std::stringstream &ss) override;
  void initialize(input_sampler sampler, int cdim, std::set<int> ireq) override;
  std::string status_report() const override;
  evaluator_ptr clone() const override;
  double complexity_penalty() const override;
  double complexity() const override;
  std::set<int> list_inputs() const override;

  void example_setup(int cdim);
};
