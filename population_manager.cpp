#include <sstream>
#include <string>

#include "population_manager.hpp"

using namespace std;

population_manager::population_manager(int popsize) : popsize(popsize) {}

string population_manager::pop_stats(string row_prefix) const {
  sstream ss;
  string sep = ",";
  for (i = 0; i < player.size(); i++) {
    ss << row_prefix << sep << player[i]->id << sep << (i + 1) << sep << player[i]->ancestors.size() << sep << player[i]->parents.size() << sep << player[i]->score << sep << player[i]->age << sep << player[i]->status_report() << endl;
  }
  fstat.close();
}

vector<agent_ptr> population_manager::topn(int n) const {
  return vector<agent_ptr>(pop.begin(), pop.begin() + n);
}

void population_manager::prepare_epoch(int epoch) {
  double ltime = log(epoch + 100);
  float winner_reward = base_game->winner_reward(epoch);
  int protected_age = ltime / 2 + 1;
  int protected_mut_age = ltime / 5 + 1;

  // fill remaining population with new samples
  int n_fill = base_population - player.size();
  cout << "Arena: starting epoch: " << epoch << ": generating " << n_fill << " new players." << endl;

  if (n_fill > 0) {
    player_buf = prepare_n([this]() { return base_game->generate_player(); }, n_fill, preplim);
    player.insert(player.end(), player_buf.begin(), player_buf.end());
  }

  if (player.size() > base_population) {
    player.resize(base_population);
  }

  // set agent selector randomness and score
  double q = 0.5 + 0.4 * sigmoid(epoch, 300);
  for (auto a : player) {
    a->last_score = a->score;
    a->set_exploration_rate(q);
  }
}

void population_manager::evolve() {
  // select nkeep best players
  sort(player.begin(), player.end(), [](agent_ptr a, agent_ptr b) -> bool { return a->score > b->score; });

  // update simple score limit
  int lim_idx = 0.67 * player.size();
  simple_score_limit = 0.1 * player[lim_idx]->simple_score + 0.9 * simple_score_limit;

  player_buf = player;
  player_buf.resize(nkeep);

  // protect children and players who's scores are still increasing
  int n_protprog = 0;
  int n_protchild = 0;
  int n_protmut = 0;
  for (int i = nkeep; i < player.size(); i++) {
    agent_ptr a = player[i];
    if (a->was_protected) player_buf.push_back(a);

    // update protection after adding agent so scores are calculated
    // before the agent is evaluated
    double delta = a->score - a->last_score;
    double limit_score = player[nkeep - 1]->score - a->score;
    bool protect_progress = delta > limit_score / 20;
    bool protect_child = a->age < protected_age;
    bool protect_mutant = a->mut_age < protected_mut_age;

    a->was_protected = protect_progress || protect_child || protect_mutant;

    n_protmut += protect_mutant;
    n_protprog += protect_progress;
    n_protchild += protect_child;
  }

  int free_spots = player.size() - player_buf.size();
  int used_spots = 0;

  cout << "Arena: keeping " << nkeep << ", protected " << n_protprog << " progressors, " << n_protmut << " mutants and " << n_protchild << " children" << endl;

  auto mate_generator = [this, nkeep, player]() -> agent_ptr {
    int idx1 = rand_int(0, nkeep - 1);
    agent_ptr parent1 = player[idx1];

    vector<agent_ptr> buf = player;
    buf.erase(buf.begin() + idx1);

    sort(buf.begin(), buf.end(), [parent1](agent_ptr a, agent_ptr b) {
      return mate_score(parent1, a) > mate_score(parent1, b);
    });

    agent_ptr parent2 = ranked_sample(buf, 0.8);

    return parent1->mate(parent2);
  };

  auto mutate_generator = [this, nkeep, player]() -> agent_ptr {
    int idx1 = rand_int(0, nkeep - 1);
    return player[idx1]->mutate();
  };

  // mating
  int n_mate = ceil(0.5 * (float)free_spots);
  int n_mutate = ceil(0.9 * (float)(free_spots - n_mate));
  int n_init = free_spots - n_mate - n_mutate;
  vector<agent_ptr> buf;

  cout << "Mating: " << n_mate << endl;
  buf = prepare_n(mate_generator, n_mate, fmax(preplim, simple_score_limit));
  player_buf.insert(player_buf.end(), buf.begin(), buf.end());

  cout << "Mutating: " << n_mutate << endl;
  buf = prepare_n(mutate_generator, n_mutate, fmax(preplim, simple_score_limit));
  player_buf.insert(player_buf.end(), buf.begin(), buf.end());
  player = player_buf;

  // age players
  for (auto x : player) {
    x->age++;
    x->mut_age++;
    x->score *= (1 - SCORE_UPDATE_RATE);
  }
}