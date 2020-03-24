#pragma once
#include <sstream>
#include <string>

#include "types.hpp"

class choice {
 public:
  float value_buf;

  virtual bool validate() = 0;
  virtual int vector_dim() = 0;
};

enum cs_schema {
  CS_RANKED,
  CS_WEIGHTED
};

struct choice_selector {
  float xrate;
  cs_schema schema;

 public:
  choice_selector(float r, cs_schema s = CS_RANKED);
  choice_ptr select(std::vector<choice_ptr> opts);
  void set_exploration_rate(float r);
  void set_schema(cs_schema s);
  std::string serialize() const;
  void deserialize(std::stringstream &ss);
};

// typedef std::function<vec(game::ptr g, int pid, choice::ptr c)> vectorizer;
// typedef std::function<std::vector<choice::ptr>(game::ptr, int)> generator;
// typedef std::function<choice::ptr(game::ptr g, agent::ptr a)> selector;

// selector ranked_selector(double q);
