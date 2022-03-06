#include "team.hpp"

#include "agent.hpp"
#include "utility.hpp"

using namespace std;

team::team(vector<agent_ptr> ps) : players(ps) {
  static int idc = 0;

  score = simple_score();
  last_rank = 0;
  rank = 0;
  was_protected = false;
  id = idc++;
}

void team::deserialize(stringstream &ss) {
  int n;
  ss >> rank >> last_rank >> was_protected >> id >> n;
  players.resize(n);
  for (auto &a : players) a = deserialize_agent(ss);
}

string team::serialize() const {
  stringstream ss;
  ss << rank << sep << last_rank << sep << was_protected << sep << id << sep << players.size() << sep;
  for (auto a : players) ss << serialize_agent(a) << sep;
  return ss.str();
}

double team::simple_score() const {
  if (players.empty()) return 0;
  return vector_sum(map<agent_ptr, double>([](agent_ptr a) { return a->simple_score; }, players)) / players.size();
}

void team::set_exploration_rate(double q) {
  for (auto a : players) a->set_exploration_rate(q);
}