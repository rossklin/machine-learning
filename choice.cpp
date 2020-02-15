#include <algorithm>

#include "agent.hpp"
#include "choice.hpp"
#include "game.hpp"
#include "utility.hpp"

using namespace std;

choice_selector::choice_selector(float r) : xrate(r){};

choice_ptr choice_selector::select(game_ptr g, agent_ptr a) {
  vector<choice_ptr> opts = g->generate_choices(a);

  // evaluate options
  for (auto opt : opts) {
    opt->value_buf = a->evaluate_choice(g->vectorize_choice(opt, a->id));
  };

  // sort options
  sort(opts.begin(), opts.end(), [](choice_ptr a, choice_ptr b) {
    return a->value_buf > b->value_buf;
  });

  // chose an option with exponential probability per rank
  while (true) {
    for (auto opt : opts) {
      if (u01() > xrate) return opt;
    }
  }
}

void choice_selector::set_exploration_rate(float r) { xrate = r; }

choice_selector_ptr choice_selector::clone() { return choice_selector_ptr(new choice_selector(xrate)); }