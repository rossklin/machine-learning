#include "team_evaluator.hpp"

#include <sstream>

#include "evaluator.hpp"
#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

team_evaluator::team_evaluator() : evaluator() {
  tag = "team";
}

team_evaluator::team_evaluator(vector<evaluator_ptr> e, int ri) : evaluator(), evals(e), role_index(ri) {
  tag = "team";
  stable = true;
}

void team_evaluator::update_stable() {
  for (auto e : evals) stable = stable && e->stable;
}

void team_evaluator::reset_memory_weights(double a) {
  for (auto e : evals) e->reset_memory_weights(a);
}

void team_evaluator::set_weights(const vec &w) {
}

vec team_evaluator::get_weights() const {
  return {};
}

// gradient of delta² - w_reg |w|
// wrt w, where delta = target - output
vec team_evaluator::gradient(vec input, double target) const {
  return {};
}

double team_evaluator::evaluate(vec x) {
  int team_idx = x[role_index];

  // sanity check
  assert(team_idx < evals.size() && team_idx >= 0);
  assert(x[role_index] == floor(x[role_index]));

  return evals[team_idx]->evaluate(x);
}

evaluator_ptr team_evaluator::update(std::vector<record> results, agent_ptr a, double &rel_change) const {
  team_evaluator::ptr buf = static_pointer_cast<team_evaluator>(clone());
  if (results.empty()) return buf;

  hm<int, vector<record>> parts;

  for (auto r : results) {
    int team_idx = r.state[role_index];
    assert(team_idx < evals.size());
    parts[team_idx].push_back(r);
  }

  rel_change = 0;
  double rc = 0;
  for (auto x : parts) {
    evaluator_ptr test = evals[x.first]->update(x.second, a, rc);
    if (test) {
      rel_change += rc;
      buf->evals[x.first] = test;
    } else {
      return NULL;
    }
  }

  rel_change /= parts.size();

  buf->update_stable();
  return buf;
}

void team_evaluator::prune(double l) {
  for (auto e : evals) e->prune(l);
  update_stable();
}

evaluator_ptr team_evaluator::mate(evaluator_ptr _partner) const {
  int n = evals.size();
  team_evaluator::ptr partner = static_pointer_cast<team_evaluator>(_partner);
  team_evaluator::ptr child(new team_evaluator(*this));
  for (int i = 0; i < n; i++) child->evals[i] = sample_one(evals)->mate(sample_one(partner->evals));
  child->update_stable();

  return child;
}

evaluator_ptr team_evaluator::mutate(evaluator::dist_category dc) const {
  if (dc == MUT_RANDOM) dc = sample_one<dist_category>({MUT_SMALL, MUT_MEDIUM, MUT_LARGE});

  team_evaluator::ptr child(new team_evaluator(*this));
  for (int i = 0; i < child->evals.size(); i++) child->evals[i] = evals[i]->mutate(dc);
  child->update_stable();
  child->mut_tag = dc;

  return child;
}

std::string team_evaluator::serialize() const {
  stringstream ss;
  ss << evaluator::serialize() << sep << evals.size() << sep << role_index << sep;
  for (auto e : evals) ss << serialize_evaluator(e) << sep;
  return ss.str();
}

void team_evaluator::deserialize(std::stringstream &ss) {
  int n;
  evaluator::deserialize(ss);
  ss >> n >> role_index;
  assert(n > 0 && n < 1e4);
  evals.resize(n);
  for (auto &e : evals) e = deserialize_evaluator(ss);
}

void team_evaluator::initialize(input_sampler sampler, int cdim, std::set<int> ireq) {
  for (auto e : evals) e->initialize(sampler, cdim, ireq);
  update_stable();
}

std::string team_evaluator::status_report() const {
  stringstream ss;

  ss << complexity() << comma
     << mut_tag << comma
     << join_string(map<evaluator_ptr, string>([](evaluator_ptr e) { return e->status_report(); }, evals), comma);

  return ss.str();
}

evaluator_ptr team_evaluator::clone() const {
  team_evaluator::ptr child(new team_evaluator(*this));
  // child->evals = map<evaluator_ptr, evaluator_ptr>([](evaluator_ptr e) { return e->clone(); }, evals);
  child->evals = map<evaluator_ptr, evaluator_ptr>(&evaluator::clone, evals);
  return child;
}

double team_evaluator::complexity() const {
  double sum = 0;
  for (auto e : evals) sum += e->complexity();
  return sum;
}

set<int> team_evaluator::list_inputs() const {
  assert(evals.size() > 0);
  set<int> common = evals.front()->list_inputs();
  for (auto e : evals) common = set_intersect(common, e->list_inputs());
  return set_union(common, {role_index});
}

void team_evaluator::add_inputs(set<int> inputs) {
  for (auto e : evals) e->add_inputs(inputs);
}