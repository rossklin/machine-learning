#pragma once

#include <memory>
#include <set>
#include <sstream>
#include <string>

#include "types.hpp"

// AGENT
class agent : public std::enable_shared_from_this<agent> {
 public:
  static int pid;

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

  agent();
  virtual agent_ptr clone() = 0;
  virtual choice_ptr select_choice(game_ptr g) = 0;
  virtual void set_exploration_rate(float r) = 0;
  virtual agent_ptr mate(agent_ptr p) = 0;
  virtual agent_ptr mutate() = 0;
  virtual double evaluate_choice(vec x) = 0;
  virtual evaluator_ptr mate_evaluator(evaluator_ptr p) = 0;
  virtual void update_evaluator(vec input, float sf_rewards) = 0;
  virtual void initialize_from_input(input_sampler s, int choice_dim) = 0;
  virtual float complexity_penalty() = 0;
  virtual bool evaluator_stability() = 0;
  virtual std::string status_report() = 0;
  virtual std::string serialize() = 0;
  virtual void deserialize(std::stringstream &s) = 0;
};

class standard_agent : public agent {
  evaluator_ptr eval;
  choice_selector_ptr csel;

 public:
  standard_agent(evaluator_ptr e, choice_selector_ptr c);
  standard_agent_ptr clone_std();
  bool evaluator_stability();

  virtual choice_ptr select_choice(game_ptr g);
  virtual void set_exploration_rate(float r);
  virtual agent_ptr mate(agent_ptr p);
  virtual agent_ptr mutate();
  virtual double evaluate_choice(vec x);
  virtual evaluator_ptr mate_evaluator(evaluator_ptr p);
  virtual void update_evaluator(vec input, float sf_rewards);
  virtual void initialize_from_input(input_sampler s, int choice_dim);
  virtual std::string status_report();
  virtual float complexity_penalty();
  virtual std::string serialize();
  virtual void deserialize(std::stringstream &s);
};