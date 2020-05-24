#pragma once

#include <memory>
#include <set>
#include <sstream>
#include <string>

#include "choice.hpp"
#include "types.hpp"

enum agent_class {
  POD_AGENT
};

struct training_stats {
  double rel_change_mean;
  double rate_successfull;
  double rate_accurate;
  double rate_optim_failed;
  double output_change;
  int n;

  training_stats();
};

std::string serialize_agent(agent_ptr a);
agent_ptr deserialize_agent(std::stringstream &ss);

class agent : public std::enable_shared_from_this<agent> {
 public:
  static int idc;

  choice_selector_ptr csel;
  evaluator_ptr eval;

  // agent classifiers
  int class_id;
  int id;
  int original_id;
  std::string label;

  // buffer variables
  int team;
  int team_index;
  int assigned_game;
  bool was_protected;
  std::vector<agent_ptr> parent_buf;  // for use in prepared player

  // stats
  double score_tmt;
  dvalue score_simple;
  dvalue score_refbot;
  dvalue score_retiree;
  std::string retiree_id;
  int rank;
  int last_rank;
  int age;
  int mut_age;

  // learning system parameters
  double future_discount;
  double w_reg;      // todo: regularization
  double mem_limit;  // todo: cap nr memories
  double mem_curve;  // todo: length of mem fade curve

  training_stats tstats;

  std::set<int> parents;
  std::set<int> ancestors;

  // constructors
  agent();
  virtual void deserialize(std::stringstream &ss);

  // duplicators
  virtual agent_ptr clone() const = 0;
  virtual agent_ptr mate(agent_ptr p) const;
  virtual agent_ptr mutate() const;

  // modifiers
  virtual void train(std::vector<std::vector<record>> records, input_sampler isam);
  virtual void set_exploration_rate(float r);
  virtual void initialize_from_input(input_sampler s, int choice_dim, std::set<int> ireq);

  // analysis
  virtual choice_ptr select_choice(game_ptr g);
  virtual double evaluate_choice(vec x) const;
  virtual bool evaluator_stability() const;
  virtual std::string status_report() const;
  virtual std::string serialize() const;
};
