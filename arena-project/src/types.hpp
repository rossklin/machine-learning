#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class agent;
class game;
class game_generator;
class evaluator;
class choice_selector;
class choice;
class population_manager;
class tournament;

typedef std::shared_ptr<agent> agent_ptr;
typedef std::shared_ptr<game> game_ptr;
typedef std::shared_ptr<game_generator> game_generator_ptr;
typedef std::shared_ptr<choice> choice_ptr;
typedef std::shared_ptr<choice_selector> choice_selector_ptr;
typedef std::shared_ptr<evaluator> evaluator_ptr;
typedef std::shared_ptr<population_manager> population_manager_ptr;
typedef std::shared_ptr<tournament> tournament_ptr;

typedef std::vector<double> vec;

struct point {
  double x;
  double y;
};

template <typename K, typename V>
using hm = std::unordered_map<K, V>;

struct option {
  vec choice;
  vec input;
  double output;
  int original_idx;
};
struct record {
  vec state;
  std::vector<option> opts;
  int selected_option;
  double reward;
  double sum_future_rewards;
};

typedef hm<int, record> record_table;
typedef hm<int, std::vector<record>> game_result;
typedef hm<int, agent_ptr> player_table;

struct t_unary {
  std::function<double(double)> f;
  std::function<double(double)> fprime;
};

struct t_binary {
  std::function<double(double, double)> f;
  std::function<double(double, double)> dfdx1;
  std::function<double(double, double)> dfdx2;
};

typedef std::function<record()> input_sampler;
typedef std::function<agent_ptr()> agent_f;
