#include "choice.hpp"

#include <algorithm>

#include "agent.hpp"
#include "game.hpp"
#include "utility.hpp"

using namespace std;

choice_selector::choice_selector(float r, cs_schema s) : xrate(r), schema(s){};

int choice_selector::select(vector<option> opts) {
  if (u01() < xrate) return rand_int(0, opts.size() - 1);

  for (int i = 0; i < opts.size(); i++) opts[i].original_idx = i;

  // todo: support for weighted selection
  // sort options
  sort(opts.begin(), opts.end(), [](option a, option b) {
    return a.output > b.output;
  });

  // chose an option with exponential probability per rank
  int rank = ranked_sample(seq(0, opts.size() - 1), 1 - xrate);

  return opts[0].original_idx;
}

void choice_selector::set_exploration_rate(float r) { xrate = r; }
void choice_selector::set_schema(cs_schema s) { schema = s; }

string choice_selector::serialize() const {
  stringstream ss;
  ss << xrate << " " << schema;
  return ss.str();
}

void choice_selector::deserialize(stringstream &ss) {
  int buf;
  ss >> xrate >> buf;
  schema = cs_schema(buf);
}