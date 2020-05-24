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

// todo: always 0 for first generation
double mate_score(agent_ptr parent1, agent_ptr x) {
  // direct descendant
  if (parent1->parents.count(x->id) || x->parents.count(parent1->id)) return -10;

  double cpenalty = 3 * sigmoid(x->eval->complexity(), 500);
  double different_ancestors = fmin(5, set_symdiff(x->ancestors, parent1->ancestors).size());
  double common_parents = set_intersect(x->parents, parent1->parents).size();

  return x->score_tmt + 3 * x->score_refbot.current + 5 * x->score_retiree.current + different_ancestors - common_parents - cpenalty;
};

void population_manager::sortpop() {
  sort(pop.begin(), pop.end(), [](agent_ptr a, agent_ptr b) -> bool { return a->score_tmt > b->score_tmt; });
  for (int i = 0; i < pop.size(); i++) pop[i]->rank = i + 1;
}

void population_manager::evolve(game_generator_ptr gg) {
  check_gg(gg);

  // pop management parameters
  int protected_age = 5;
  int protected_mut_age = 3;

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
  simple_score_limit = 0.1 * pop[lim_idx]->score_simple.current + 0.9 * simple_score_limit;

  auto cond_drop = [](agent_ptr a) {
    bool has_data = a->tstats.n > 100;
    bool slow_optim = a->tstats.rate_successfull < 0.5;
    bool slow_update = a->tstats.output_change < 1e-5;
    bool slow = slow_optim || slow_update;
    bool stalled = a->tstats.output_change < 1e-6 || a->tstats.rate_successfull < 0.2;
    bool old = a->age > 1000;
    bool ancient = a->age > 5000;

    if (ancient || (has_data && (stalled || (old && slow)))) {
      cout << "Dropping agent with stats: " << a->tstats.rate_successfull << ", " << a->age << ", " << a->tstats.output_change << endl;
      return true;
    } else {
      return false;
    }
  };

  vector<agent_ptr> player_buf;
  int n_retire = 0;
  for (int i = 0; i < nkeep; i++) {
    agent_ptr a = pop[i];
    if (!cond_drop(a)) {
      player_buf.push_back(a);
    } else if (i < 3) {
      n_retire++;
      retirement.insert(retirement.begin(), a);
    }
  }
  if (retirement.size() > 100) retirement.resize(100);
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
    if (a->rank < a->last_rank - 1) {
      a->was_protected = true;
      player_buf.push_back(a);
      n_protprog++;
    }
  }

  int free_spots = popsize - player_buf.size();

  // trial between population and retirement
  int n_trial = 10;
  int n_win = 0;
  float qtrial = 0.2;
  if (retirement.size() && player_buf.size()) {
    cout << "Playing retirement trials" << endl;
    for (int i = 0; i < n_trial; i++) {
      agent_ptr a = player_buf.front();
      agent_ptr b = retirement.back();
      a->set_exploration_rate(0.05);
      b->set_exploration_rate(0.05);
      game_ptr g = gg->generate_starting_state(gg->make_teams({a, b}));
      g->play(1);
      n_win += g->winner == a->team;
    }

    qtrial = (n_win + 2 - (retirement.size() > 1)) / (float)n_trial;
    cout << "Qtrial: " << qtrial << endl;
  }

  cout << "PM: keeping " << result_keep << ", retiring " << n_retire << ", dropping " << (n_drop - n_retire) << " , protected " << n_protprog << " progressors" << endl;
  cout << "Retiree count: " << retirement.size() << endl;

  auto mate_generator = [this, nkeep, qtrial]() -> agent_ptr {
    agent_ptr parent1;
    vector<agent_ptr> buf = pop;

    if (retirement.size() && u01() > qtrial) {
      // use a retired agent as parent
      parent1 = sample_one(retirement);
    } else {
      // use a top agent from population as parent
      int idx1 = rand_int(0, nkeep - 1);
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

  // // post processing
  // for (auto x : pop) {
  //   x->score_tmt *= 0.999;

  //   // Only allow pruning when creating a new agent since it will screw with memory index mapping
  //   // x->eval->prune();
  // }

  // sort population again
  sortpop();
}