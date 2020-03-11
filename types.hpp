#pragma once

#include <functional>
#include <memory>
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
typedef std::shared_ptr<evaluator> evaluator_ptr;
typedef std::shared_ptr<population_manager> population_manager_ptr;
typedef std::shared_ptr<tournament> tournament_ptr;

typedef std::vector<double> vec;
typedef std::function<vec()> input_sampler;
typedef std::function<agent_ptr()> agent_f;

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
