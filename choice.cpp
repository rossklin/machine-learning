#include <algorithm>

#include "agent.hpp"
#include "choice.hpp"
#include "game.hpp"
#include "utility.hpp"

using namespace std;

choice_selector::choice_selector(float r, cs_schema s) : xrate(r), schema(s){};

choice_ptr choice_selector::select(vector<choice_ptr> opts) {
  // todo: support for weighted selection
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
void choice_selector::set_schema(cs_schema s) { schema = s; }
