#pragma once

#include <memory>
#include <set>
#include <string>

#include "choice.hpp"
#include "types.hpp"

class agent : public std::enable_shared_from_this<agent> {
 public:
  static int idc;

  choice_selector csel;
  evaluator_ptr eval;

  int team;
  int id;
  int team_index;
  int assigned_game;
  double score;
  double last_score;
  double simple_score;
  bool was_protected;
  int age;
  int mut_age;
  std::string label;
  std::set<int> parents;
  std::set<int> ancestors;

  // constructors
  agent();
  virtual void deserialize(std::string s);

  // duplicators
  virtual agent_ptr clone() const = 0;
  virtual agent_ptr mate(agent_ptr p) const;
  virtual agent_ptr mutate() const;

  // modifiers
  virtual void train(std::vector<record> records);
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
