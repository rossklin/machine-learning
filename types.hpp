#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

class game;
class agent;
class choice;
class evaluator;
class choice_selector;
class standard_agent;
class population_manager;
class game_generator;
class tournament;

typedef std::shared_ptr<game> game_ptr;
typedef std::shared_ptr<agent> agent_ptr;
typedef std::shared_ptr<choice> choice_ptr;
typedef std::shared_ptr<evaluator> evaluator_ptr;
typedef std::shared_ptr<choice_selector> choice_selector_ptr;
typedef std::shared_ptr<standard_agent> standard_agent_ptr;
typedef std::shared_ptr<population_manager> population_manager_ptr;
typedef std::shared_ptr<game_generator> game_generator_ptr;
typedef std::shared_ptr<tournament> tournament_ptr;

typedef std::vector<double> vec;
typedef std::function<vec()> input_sampler;

struct point {
  double x;
  double y;
};

template <typename K, typename V>
using hm = std::unordered_map<K, V>;

struct record {
  vec input;
  double output;
  double reward;
  double reward_simple;
  double sum_future_rewards;
};

typedef hm<int, record> record_table;

struct t_unary {
  std::function<double(double)> f;
  std::function<double(double)> fprime;
};

struct t_binary {
  std::function<double(double, double)> f;
  std::function<double(double, double)> dfdx1;
  std::function<double(double, double)> dfdx2;
};
