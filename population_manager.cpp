#include "population_manager.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "agent.hpp"
#include "evaluator.hpp"
#include "game_generator.hpp"
#include "utility.hpp"

using namespace std;

population_manager::population_manager(int popsize, agent_f gen, float plim) : popsize(popsize), gen(gen), preplim(plim) {
  simple_score_limit = 0.2;
  assert(popsize >= 8);
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

double mate_score(agent_ptr parent1, agent_ptr x) {
  // direct descendant
  if (parent1->parents.count(x->id) || x->parents.count(parent1->id)) return 0;

  double different_ancestors = set_symdiff(x->ancestors, parent1->ancestors).size();
  double common_parents = set_intersect(x->parents, parent1->parents).size();

  return x->score + different_ancestors - common_parents;
};

void population_manager::sortpop() {
  sort(pop.begin(), pop.end(), [](agent_ptr a, agent_ptr b) -> bool { return a->score > b->score; });
  for (int i = 0; i < pop.size(); i++) pop[i]->rank = i + 1;
}

void population_manager::evolve(game_generator_ptr gg) {
  check_gg(gg);

  // pop management parameters
  int protected_age = 5;
  int protected_mut_age = 3;
  float score_update_rate = 0.03;

  // remove unstable players
  for (int i = 0; i < pop.size(); i++) {
    if (!pop[i]->evaluator_stability()) {
      cout << "PM: erasing unstable player " << pop[i]->id << endl;
      pop.erase(pop.begin() + i--);
    }
  }

  if (pop.size() < 2) {
    // can't evolve with less than two agents
    cout << "PM: not enough agents to evolve!" << endl;
    return;
  }

  // require at least 2 parents
  int nkeep = max((int)pop.size() / 4, 2);

  // select nkeep best players
  sortpop();

  // update simple score limit
  int lim_idx = 0.8 * pop.size();
  simple_score_limit = 0.1 * pop[lim_idx]->simple_score + 0.9 * simple_score_limit;

  auto cond_drop = [](agent_ptr a) {
    return a->tstats.rate_successfull < 0.5;
  };

  vector<agent_ptr> player_buf;
  for (int i = 0; i < nkeep; i++) {
    agent_ptr a = pop[i];
    if (!cond_drop(a)) {
      player_buf.push_back(a);
    } else if (i < 3) {
      retirement.insert(retirement.begin(), a);
    }
  }
  while (retirement.size() > 100) retirement.pop_back();
  int n_drop = nkeep - player_buf.size();
  int result_keep = player_buf.size();

  // protect children and players who's scores are still increasing
  int n_protprog = 0;
  // int n_protchild = 0;
  // int n_protmut = 0;
  for (int i = nkeep; i < pop.size(); i++) {
    agent_ptr a = pop[i];

    // drop agent if it is no longer successfully updating
    if (cond_drop(a)) {
      n_drop++;
      continue;
    }

    // update protection after adding agent so scores are calculated
    // before the agent is evaluated
    if (a->rank < a->last_rank - 2) {
      a->was_protected = true;
      player_buf.push_back(a);
      n_protprog++;
    }
  }

  int free_spots = popsize - player_buf.size();

  cout << "PM: keeping " << result_keep << ", dropping " << n_drop << " , protected " << n_protprog << " progressors" << endl;

  auto mate_generator = [this, nkeep]() -> agent_ptr {
    int idx1 = rand_int(0, nkeep - 1);
    agent_ptr parent1 = pop[idx1];

    vector<agent_ptr> buf = pop;
    buf.erase(buf.begin() + idx1);

    sort(buf.begin(), buf.end(), [parent1](agent_ptr a, agent_ptr b) {
      return mate_score(parent1, a) > mate_score(parent1, b);
    });

    agent_ptr parent2 = ranked_sample(buf, 0.8);
    agent_ptr child = parent1->mate(parent2);

    child->parent_buf = {parent1, parent2};
    if (retirement.size() > 0) child->parent_buf.push_back(sample_one(retirement));
    return child;
  };

  auto mutate_generator = [this, nkeep]() -> agent_ptr {
    int idx1 = rand_int(0, nkeep - 1);
    agent_ptr child = pop[idx1]->mutate();
    child->parent_buf = {pop[idx1]};
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

  // post processing
  for (auto x : pop) {
    x->score *= (1 - score_update_rate);
    x->eval->prune();
  }

  // sort population again
  sortpop();
}