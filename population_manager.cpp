#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "agent.hpp"
#include "game_generator.hpp"
#include "population_manager.hpp"
#include "utility.hpp"

using namespace std;

population_manager::population_manager(int popsize, agent_f gen, float plim) : popsize(popsize), gen(gen), preplim(plim) {
  simple_score_limit = 0.2;
  assert(popsize >= 8);
}

string population_manager::pop_stats(string row_prefix) const {
  stringstream ss;
  string sep = ",";
  for (int i = 0; i < pop.size(); i++) {
    ss << row_prefix << sep << pop[i]->id << sep << (i + 1) << sep << pop[i]->ancestors.size() << sep << pop[i]->parents.size() << sep << pop[i]->score << sep << pop[i]->age << sep << pop[i]->status_report() << endl;
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

  double ltime = log(epoch + 100);
  int protected_age = ltime / 2 + 1;
  int protected_mut_age = ltime / 5 + 1;

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
    a->last_score = a->score;
    a->set_exploration_rate(q);
  }
}

double mate_score(agent_ptr parent1, agent_ptr x) {
  // x is protected so score is not reliable
  if (x->was_protected) return 0;

  // direct descendant
  if (parent1->parents.count(x->id) || x->parents.count(parent1->id)) return 0;

  double different_ancestors = set_symdiff(x->ancestors, parent1->ancestors).size();
  double common_parents = set_intersect(x->parents, parent1->parents).size();

  return x->score + different_ancestors - common_parents;
};

void population_manager::evolve(game_generator_ptr gg) {
  check_gg(gg);

  // pop management parameters
  int protected_age = 3;
  int protected_mut_age = 3;
  float score_update_rate = 0.05;

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
  sort(pop.begin(), pop.end(), [](agent_ptr a, agent_ptr b) -> bool { return a->score > b->score; });

  // update simple score limit
  int lim_idx = 0.67 * pop.size();
  simple_score_limit = 0.1 * pop[lim_idx]->simple_score + 0.9 * simple_score_limit;

  auto player_buf = pop;
  player_buf.resize(nkeep);

  // protect children and players who's scores are still increasing
  int n_protprog = 0;
  int n_protchild = 0;
  int n_protmut = 0;
  for (int i = nkeep; i < pop.size(); i++) {
    agent_ptr a = pop[i];
    if (a->was_protected) player_buf.push_back(a);

    // update protection after adding agent so scores are calculated
    // before the agent is evaluated
    double delta = a->score - a->last_score;
    double limit_score = pop[nkeep - 1]->score - a->score;
    bool protect_progress = delta > limit_score / 20;
    bool protect_child = a->age <= protected_age;
    bool protect_mutant = a->mut_age <= protected_mut_age;

    a->was_protected = protect_progress || protect_child || protect_mutant;

    n_protmut += protect_mutant;
    n_protprog += protect_progress;
    n_protchild += protect_child;
  }

  int free_spots = popsize - player_buf.size();
  int used_spots = 0;

  cout << "PM: keeping " << nkeep << ", protected " << n_protprog << " progressors, " << n_protmut << " mutants and " << n_protchild << " children" << endl;

  auto mate_generator = [this, nkeep]() -> agent_ptr {
    int idx1 = rand_int(0, nkeep - 1);
    agent_ptr parent1 = pop[idx1];

    vector<agent_ptr> buf = pop;
    buf.erase(buf.begin() + idx1);

    sort(buf.begin(), buf.end(), [parent1](agent_ptr a, agent_ptr b) {
      return mate_score(parent1, a) > mate_score(parent1, b);
    });

    agent_ptr parent2 = ranked_sample(buf, 0.8);

    return parent1->mate(parent2);
  };

  auto mutate_generator = [this, nkeep]() -> agent_ptr {
    int idx1 = rand_int(0, nkeep - 1);
    return pop[idx1]->mutate();
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

  // age players
  for (auto x : pop) {
    x->age++;
    x->mut_age++;
    x->score *= (1 - score_update_rate);
  }
}