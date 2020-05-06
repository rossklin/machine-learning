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
  double rel_change_max;
  double rate_successfull;
  double rate_accurate;
  double rate_zero;
  double rate_correct_sign;
  double rate_optim_failed;
  double output_change;

  training_stats();
};

std::string serialize_agent(agent_ptr a);
agent_ptr deserialize_agent(std::stringstream &ss);

class agent : public std::enable_shared_from_this<agent> {
 public:
  static int idc;

  choice_selector_ptr csel;
  evaluator_ptr eval;

  int class_id;
  int team;
  int id;
  int original_id;
  int team_index;
  int assigned_game;
  double score;
  int last_rank;
  int rank;
  double simple_score;
  bool was_protected;
  int age;
  int mut_age;

  training_stats tstats;

  std::string label;
  std::set<int> parents;
  std::set<int> ancestors;
  std::vector<agent_ptr> parent_buf;  // for use in prepared player

  // constructors
  agent();
  virtual void deserialize(std::stringstream &ss);

  // duplicators
  virtual agent_ptr clone() const = 0;
  virtual agent_ptr mate(agent_ptr p) const;
  virtual agent_ptr mutate() const;

  // modifiers
  virtual void train(std::vector<record> records, input_sampler isam);
  virtual void set_exploration_rate(float r);
  virtual void initialize_from_input(input_sampler s, int choice_dim);

  // analysis
  virtual choice_ptr select_choice(game_ptr g);
  virtual double evaluate_choice(vec x) const;
  virtual float complexity_penalty() const;
  virtual bool evaluator_stability() const;
  virtual std::string status_report() const;
  virtual std::string serialize() const;
};
