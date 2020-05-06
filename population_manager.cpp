#include "population_manager.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "agent.hpp"
#include "evaluator.hpp"
#include "game.hpp"
#include "game_generator.hpp"
#include "team.hpp"
#include "utility.hpp"

using namespace std;

population_manager::population_manager(int popsize, int ppt, agent_f gen, float plim) : popsize(popsize), ppt(ppt), gen(gen), preplim(plim) {
  simple_score_limit = 0.2;
  assert(popsize >= 8);
}

vector<team> load_pop(stringstream &ss) {
  int n;
  vector<team> pop;

  ss >> n;
  assert(n >= 0 && n < 1e4);
  pop.resize(n);
  for (auto &t : pop) t.deserialize(ss);

  return pop;
}

void save_pop(stringstream &ss, vector<team> pop) {
  ss << pop.size() << sep;
  for (auto t : pop) ss << t.serialize() << sep;
}

string population_manager::serialize() const {
  stringstream ss;

  ss << preplim << sep << simple_score_limit << sep << ppt << sep;

  save_pop(ss, pop);
  save_pop(ss, retirement);

  return ss.str();
}

void population_manager::deserialize(stringstream &ss) {
  ss >> preplim >> simple_score_limit >> ppt;
  pop = load_pop(ss);
  retirement = load_pop(ss);
}

string population_manager::pop_stats(string row_prefix) const {
  stringstream ss;
  for (int i = 0; i < pop.size(); i++) {
    team t = pop[i];
    for (auto a : t.players) {
      ss << row_prefix << comma << (i + 1) << comma << t.id << comma << a->status_report() << endl;
    }
  }
  return ss.str();
}

vector<team> population_manager::topn(int n) const {
  assert(n <= pop.size());
  return vector<team>(pop.begin(), pop.begin() + n);
}

void population_manager::check_gg(game_generator_ptr gg) const {
  assert(popsize == gg->nr_of_teams);
}

void population_manager::prepare_epoch(int epoch, game_generator_ptr gg) {
  check_gg(gg);

  // fill remaining population with new samples
  int n_fill = popsize - pop.size();

  if (n_fill > 0) {
    cout << "PM: starting epoch: " << epoch << ": generating " << n_fill << " new teams." << endl;
    auto team_buf = gg->prepare_n(gen, n_fill, preplim);
    pop.insert(pop.end(), team_buf.begin(), team_buf.end());
  } else {
    cout << "PM: starting epoch: " << epoch << ": population already full." << endl;
  }

  if (pop.size() > popsize) {
    pop.resize(popsize);
  }

  // set agent selector randomness and score
  double q = 0.5 - 0.4 * sigmoid(epoch, 100);
  for (auto t : pop) {
    t.last_rank = t.rank;
    for (auto a : t.players) a->set_exploration_rate(q);
  }

  sortpop();
}

double mate_score(agent_ptr parent1, agent_ptr x) {
  // direct descendant
  if (parent1->parents.count(x->id) || x->parents.count(parent1->id)) return 0;

  double different_ancestors = set_symdiff(x->ancestors, parent1->ancestors).size();
  double common_parents = set_intersect(x->parents, parent1->parents).size();

  return x->team_score + different_ancestors - common_parents;
};

void population_manager::sortpop() {
  sort(pop.begin(), pop.end(), [](team a, team b) -> bool { return a.score > b.score; });
  for (int i = 0; i < pop.size(); i++) pop[i].rank = i + 1;
}

void population_manager::evolve(game_generator_ptr gg) {
  check_gg(gg);

  // pop management parameters
  int protected_age = 5;
  int protected_mut_age = 3;
  float score_update_rate = 0.03;

  // remove unstable players
  for (int i = 0; i < pop.size(); i++) {
    if (vector_any(map<agent_ptr, bool>([](agent_ptr a) { return !a->evaluator_stability(); }, pop[i].players))) {
      cout << "PM: erasing unstable team " << pop[i].id << endl;
      pop.erase(pop.begin() + i--);
    }
  }

  if (pop.size() < 2) {
    // can't evolve with less than two agents
    cout << "PM: not enough teams to evolve!" << endl;
    return;
  }

  // require at least 2 parents
  int nkeep = max((int)pop.size() / 4, 2);

  // select nkeep best players
  sortpop();

  // update simple score limit
  int lim_idx = 0.8 * pop.size();
  simple_score_limit = 0.1 * pop[lim_idx].simple_score() + 0.9 * simple_score_limit;

  auto cond_drop = [](agent_ptr a) {
    return a->tstats.rate_successfull < 0.1;
  };

  vector<team> team_buf;
  for (int i = 0; i < nkeep; i++) {
    team t = pop[i];
    if (!vector_any(map<agent_ptr, bool>(cond_drop, t.players))) {
      team_buf.push_back(t);
    } else if (i < 3) {
      retirement.insert(retirement.begin(), t);
    }
  }
  if (retirement.size() > 100) retirement.resize(100);
  int n_drop = nkeep - team_buf.size();
  int result_keep = team_buf.size();

  // protect children and players who's scores are still increasing
  int n_protprog = 0;
  // int n_protchild = 0;
  // int n_protmut = 0;
  for (int i = nkeep; i < pop.size(); i++) {
    team t = pop[i];

    // drop agent if it is no longer successfully updating
    if (vector_any(map<agent_ptr, bool>(cond_drop, t.players))) {
      n_drop++;
      continue;
    }

    // update protection after adding agent so scores are calculated
    // before the agent is evaluated
    if (t.rank < t.last_rank - 1) {
      t.was_protected = true;
      team_buf.push_back(t);
      n_protprog++;
    }
  }

  int free_spots = popsize - team_buf.size();

  // trial between population and retirement
  int n_trial = 10;
  int n_win = 0;
  if (retirement.size()) {
    for (int i = 0; i < n_trial; i++) {
      team a = team_buf.front();
      team b = retirement.back();
      a.set_exploration_rate(0.05);
      b.set_exploration_rate(0.05);
      game_ptr g = gg->generate_starting_state({a, b});
      g->play(1);
      n_win += g->winner == a.id;
    }
  }
  float qtrial = (n_win + 2 - (retirement.size() > 1)) / (float)n_trial;

  cout << "PM: keeping " << result_keep << ", retiring " << n_drop << " , protected " << n_protprog << " progressors" << endl;
  cout << "Retirement trials: won " << n_win << " of " << n_trial << endl;

  auto mate_generator = [this, nkeep, qtrial]() -> agent_ptr {
    agent_ptr parent1;
    vector<agent_ptr> buf = vector_merge(map<team, vector<agent_ptr>>([](team t) { return t.players; }, pop));

    if (retirement.size() && u01() > qtrial) {
      // use a retired agent as parent
      parent1 = sample_one(sample_one(retirement).players);
    } else {
      // use a top agent from population as parent
      int idx1 = rand_int(0, ppt * nkeep - 1);
      parent1 = buf[idx1];
      buf.erase(buf.begin() + idx1);
    }

    sort(buf.begin(), buf.end(), [parent1](agent_ptr a, agent_ptr b) {
      return mate_score(parent1, a) > mate_score(parent1, b);
    });

    agent_ptr parent2 = ranked_sample(buf, 0.5);
    agent_ptr child = parent1->mate(parent2);

    child->parent_buf = {parent1, parent2};
    if (retirement.size() > 0) child->parent_buf.push_back(sample_one(sample_one(retirement).players));
    return child;
  };

  auto mutate_generator = [this, nkeep]() -> agent_ptr {
    int idx1 = rand_int(0, nkeep - 1);
    agent_ptr parent = sample_one(pop[idx1].players);
    agent_ptr child = parent->mutate();
    child->parent_buf = {parent};
    if (retirement.size() > 0) child->parent_buf.push_back(sample_one(sample_one(retirement).players));
    return child;
  };

  // mating
  int n_mate = ceil(0.5 * (float)free_spots);
  int n_mutate = ceil(0.9 * (float)(free_spots - n_mate));
  int n_init = free_spots - n_mate - n_mutate;
  vector<team> buf;

  cout << "Mating: " << n_mate << endl;
  buf = gg->prepare_n(mate_generator, n_mate, fmax(preplim, simple_score_limit));
  team_buf.insert(team_buf.end(), buf.begin(), buf.end());

  cout << "Mutating: " << n_mutate << endl;
  buf = gg->prepare_n(mutate_generator, n_mutate, fmax(preplim, simple_score_limit));
  team_buf.insert(team_buf.end(), buf.begin(), buf.end());
  pop = team_buf;

  // post processing
  for (auto x : pop) {
    x.score *= (1 - score_update_rate);
    for (auto a : x.players) a->eval->prune();
  }

  // sort population again
  sortpop();
}