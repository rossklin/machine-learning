#include "population_manager.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "agent.hpp"
#include "evaluator.hpp"
#include "game.hpp"
#include "game_generator.hpp"
#include "utility.hpp"

using namespace std;

population_manager::population_manager(int popsize, agent_f gen, float plim) : popsize(popsize), gen(gen), preplim(plim) {
  simple_score_limit = 0.2;
  assert(popsize >= 8);
  refbot = 0;
}

vector<agent_ptr> load_pop(stringstream &ss) {
  int n;
  vector<agent_ptr> pop;

  ss >> n;
  assert(n >= 0 && n < 1e4);
  pop.resize(n);
  for (auto &a : pop) a = deserialize_agent(ss);

  return pop;
}

void save_pop(stringstream &ss, vector<agent_ptr> pop) {
  ss << pop.size() << sep;
  for (auto a : pop) ss << serialize_agent(a) << sep;
}

string population_manager::serialize() const {
  stringstream ss;

  ss << preplim << sep << simple_score_limit << sep;

  save_pop(ss, pop);
  save_pop(ss, retirement);

  return ss.str();
}

void population_manager::deserialize(stringstream &ss) {
  ss >> preplim >> simple_score_limit;
  pop = load_pop(ss);
  retirement = load_pop(ss);
}

string population_manager::pop_stats(string row_prefix) const {
  stringstream ss;
  string comma = ",";
  for (int i = 0; i < pop.size(); i++) {
    ss << row_prefix << comma << (i + 1) << comma << pop[i]->status_report() << endl;
  }
  return ss.str();
}

vector<agent_ptr> population_manager::topn(int n) const {
  assert(n <= pop.size());
  return vector<agent_ptr>(pop.begin(), pop.begin() + n);
}

void population_manager::check_gg(game_generator_ptr gg) const {
  assert(popsize % gg->nr_of_teams == 0);
}

void population_manager::prepare_epoch(int epoch, game_generator_ptr gg) {
  check_gg(gg);

  // fill remaining population with new samples
  int n_fill = popsize - pop.size();

  if (n_fill > 0) {
    cout << "PM: starting epoch: " << epoch << ": generating " << n_fill << " new players." << endl;
    auto player_buf = gg->prepare_n(gen, n_fill, preplim);
    pop.insert(pop.end(), player_buf.begin(), player_buf.end());
  } else {
    cout << "PM: starting epoch: " << epoch << ": population already full." << endl;
  }

  if (pop.size() > popsize) {
    pop.resize(popsize);
  }

  // set agent selector randomness and score
  double q = 0.5 - 0.4 * sigmoid(epoch, 100);
  for (auto a : pop) {
    a->last_rank = a->rank;
    a->set_exploration_rate(q);
  }

  sortpop();
}

// todo: always 0 for first generation
double mate_score(agent_ptr parent1, agent_ptr x) {
  // direct descendant
  if (parent1->parents.count(x->id) || x->parents.count(parent1->id)) return -10;

  double cpenalty = 3 * sigmoid(x->eval->complexity(), 500);
  double different_ancestors = fmin(5, set_symdiff(x->ancestors, parent1->ancestors).size());
  double common_parents = set_intersect(x->parents, parent1->parents).size();

  return x->score_tmt.value_ma + 3 * x->score_refbot.value_ma + different_ancestors - common_parents - cpenalty;
};

void population_manager::sortpop() {
  sort(pop.begin(), pop.end(), [](agent_ptr a, agent_ptr b) -> bool { return a->score_tmt.value_ma > b->score_tmt.value_ma; });
  for (int i = 0; i < pop.size(); i++) pop[i]->rank = i + 1;
}

void population_manager::evolve(game_generator_ptr gg) {
  check_gg(gg);

  // calculate quantile of refbot score and update the refbot if necessary
  sort(pop.begin(), pop.end(), [](agent_ptr a, agent_ptr b) -> bool { return a->score_refbot.value_ma > b->score_refbot.value_ma; });
  double score_refbot_q90 = pop[0.1 * pop.size()]->score_refbot.value_ma;

  if (retirement.size() > 0 && score_refbot_q90 > 0.95) {
    // time to switch to a stronger refbot
    int idx = min<int>(9, retirement.size() - 1);
    if (refbot) {
      // guarantee selecting a newer refbot than the last one
      for (int i = 0; i <= idx; i++) {
        if (retirement[i]->id == refbot->id) {
          idx = i - 1;
        }
      }
    }

    if (idx >= 0) {
      refbot = retirement[idx];

      for (auto a : pop) {
        // Restart statistics for score_refbot
        a->score_refbot = dvalue();

        // Update memory weights:
        // since current top players win consistently against old refbot,
        // and are expeccted to win 50% against new refbot,
        // we heuristically expect all agents to win half as often against the new refbot,
        // therefore old memory weights should be halved to compare fairly to new memories.
        a->eval->reset_memory_weights(0.5);
      }
    }
  }

  // remove unstable players
  for (int i = 0; i < pop.size(); i++) {
    if (!pop[i]->evaluator_stability()) {
      cout << "PM: erasing unstable player " << pop[i]->id << endl;
      pop.erase(pop.begin() + i--);
    }
  }

  // require at least 2 parents
  int nkeep = max((int)pop.size() / 4, 2);

  // select nkeep best players
  sortpop();

  // update simple score limit
  int lim_idx = 0.8 * pop.size();
  simple_score_limit = 0.1 * pop[lim_idx]->score_simple.current + 0.9 * simple_score_limit;

  // determine which agents to keep
  auto agent_evaluator = [](agent_ptr a) -> int {
    if (a->tstats.n < 10) return 1;  // All agents must have played at least one tournament before being evaluated

    int stalled = a->tstats.output_change < 1e-6 || a->tstats.rate_successfull < 0.2;
    int ancient = a->age > 500;

    int improve_tmt = a->score_tmt.diff_ma > 0.01 || a->rank < a->last_rank;
    int improve_perf = a->score_refbot.diff_ma > 0.01;
    int improve_speed = a->score_simple.diff_ma > 0.01;

    return improve_perf + improve_speed + improve_tmt - stalled - ancient;
  };

  vector<agent_ptr> player_buf;
  int n_retire = 0;
  int n_drop = 0;
  for (int i = 0; i < pop.size(); i++) {
    agent_ptr a = pop[i];
    if (agent_evaluator(a) > 0) {
      player_buf.push_back(a);
    } else if (i < nkeep) {
      n_retire++;
      retirement.insert(retirement.begin(), a);
    } else {
      n_drop++;
    }
  }
  if (retirement.size() > 100) retirement.resize(100);
  int result_keep = player_buf.size();
  int max_idx = min(nkeep, result_keep);
  int free_spots = popsize - player_buf.size();

  if (player_buf.empty() || pop.size() < 2) {
    // can't evolve with less than two agents
    cout << "PM: not enough agents to evolve!" << endl;
    pop = player_buf;
    return;
  }

  cout << "PM: keeping " << result_keep << ", retiring " << n_retire << ", dropping " << n_drop << endl;
  cout << "Retiree count: " << retirement.size() << endl;

  auto mate_generator = [this, max_idx]() -> agent_ptr {
    agent_ptr parent1;
    vector<agent_ptr> buf = pop;

    if (retirement.size() && u01() > 0.5) {
      // use a retired agent as parent
      parent1 = sample_one(retirement);
    } else {
      // use a top agent from population as parent
      int idx1 = rand_int(0, max_idx - 1);
      parent1 = pop[idx1];
      buf.erase(buf.begin() + idx1);
    }

    sort(buf.begin(), buf.end(), [parent1](agent_ptr a, agent_ptr b) {
      return mate_score(parent1, a) > mate_score(parent1, b);
    });

    agent_ptr parent2 = ranked_sample(buf, 0.8);
    agent_ptr child = parent1->mate(parent2);

    child->parent_buf = {parent1, parent2};
    if (retirement.size() > 0) child->parent_buf.push_back(sample_one(retirement));

    return child;
  };

  auto mutate_generator = [this]() -> agent_ptr {
    agent_ptr parent = ranked_sample(pop, 0.33);
    agent_ptr child = parent->mutate();
    child->parent_buf = {parent};
    if (retirement.size() > 0) child->parent_buf.push_back(sample_one(retirement));

    return child;
  };

  // mating
  int n_mate = ceil(0.5 * (float)free_spots);
  int n_mutate = ceil(0.9 * (float)(free_spots - n_mate));
  int n_init = free_spots - n_mate - n_mutate;
  vector<agent_ptr> buf;

  cout << "Mating: " << n_mate << endl;
  buf = gg->prepare_n(mate_generator, n_mate, fmax(preplim, simple_score_limit));
  player_buf.insert(player_buf.end(), buf.begin(), buf.end());

  cout << "Mutating: " << n_mutate << endl;
  buf = gg->prepare_n(mutate_generator, n_mutate, fmax(preplim, simple_score_limit));
  player_buf.insert(player_buf.end(), buf.begin(), buf.end());
  pop = player_buf;

  // sort population again
  sortpop();
}