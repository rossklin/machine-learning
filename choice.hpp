#pragma once

class choice {
 public:
  virtual bool validate() = 0;
  virtual int vector_dim() = 0;
};

class choice_selector {
  float xrate;

 public:
  choice_selector(float r);
  virtual choice_ptr select(game_ptr g, agent_ptr a);
  virtual void set_exploration_rate(float r);
  virtual choice_selector_ptr clone();
};

// typedef std::function<vec(game::ptr g, int pid, choice::ptr c)> vectorizer;
// typedef std::function<std::vector<choice::ptr>(game::ptr, int)> generator;
// typedef std::function<choice::ptr(game::ptr g, agent::ptr a)> selector;

// selector ranked_selector(double q);
