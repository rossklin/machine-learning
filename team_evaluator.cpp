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
  set_learning_rate(fabs(rnorm(0, 0.5)));
  stable = true;
}

void team_evaluator::update_stable() {
  for (auto e : evals) stable = stable && e->stable;
}

void team_evaluator::set_learning_rate(double r) {
  learning_rate = r;
  for (auto e : evals) e->set_learning_rate(r);
}

void team_evaluator::set_weights(const vec &w) {
  // int offset = 0;
  // for (auto e : evals) {
  //   int n = e->get_weights().size();
  //   vec buf(w.begin() + offset, w.begin() + offset + n);
  //   e->set_weights(buf);
  //   offset += n;
  // }
}

vec team_evaluator::get_weights() const {
  // vec res;
  // for (auto e : evals) res = vec_append(res, e->get_weights());
  // return res;
  return {};
}

// gradient of delta² - w_reg |w|
// wrt w, where delta = target - output
vec team_evaluator::gradient(vec input, double target, double w_reg) const {
  // int team_idx = input[role_index];
  // assert(team_idx < evals.size());
  // vec res;
  // for (int i = 0; i < evals.size(); i++) {
  //   int n = evals[i]->get_weights().size();
  //   if (i == team_idx) {
  //     res = vec_append(res, evals[i]->gradient(input, target, w_reg));
  //   } else {
  //     res = vec_append(res, vec(n, 0));
  //   }
  // }

  // return res;
  return {};
}

double team_evaluator::evaluate(vec x) {
  int team_idx = x[role_index];
  assert(team_idx < evals.size());
  return evals[team_idx]->evaluate(x);
}

bool team_evaluator::update(std::vector<record> results, agent_ptr a, double &rel_change) {
  if (results.empty()) return true;

  hm<int, vector<record>> parts;

  for (auto r : results) {
    int team_idx = r.input[role_index];
    assert(team_idx < evals.size());
    parts[team_idx].push_back(r);
  }

  rel_change = 0;
  double rc = 0;
  for (auto x : parts) {
    bool test = evals[x.first]->update(x.second, a, rc);
    rel_change += rc;
    if (!test) return false;
  }

  rel_change /= parts.size();

  update_stable();
  return stable;
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
  child->set_learning_rate(0.5 * (learning_rate + _partner->learning_rate));
  child->update_stable();

  return child;
}

evaluator_ptr team_evaluator::mutate(evaluator::dist_category dc) const {
  if (dc == MUT_RANDOM) dc = sample_one<dist_category>({MUT_SMALL, MUT_MEDIUM, MUT_LARGE});

  team_evaluator::ptr child(new team_evaluator(*this));
  for (int i = 0; i < child->evals.size(); i++) child->evals[i] = evals[i]->mutate(dc);
  child->set_learning_rate(fmax(learning_rate + rnorm(0, 1e-2), 1e-5));
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
  set_learning_rate(learning_rate);
}

void team_evaluator::initialize(input_sampler sampler, int cdim, std::set<int> ireq) {
  for (auto e : evals) e->initialize(sampler, cdim, ireq);
  set_learning_rate(learning_rate);
  update_stable();
}

std::string team_evaluator::status_report() const {
  // string tree_sub = join_string(
  //     map<evaluator_ptr, string>(
  //         [](evaluator_ptr e) {
  //           return e->tag == "tree" ? to_string(static_pointer_cast<tree_evaluator>(e)->gamma) : "NA";
  //         },
  //         evals),
  //     comma);
  return to_string(complexity()) + comma + to_string(learning_rate) + comma + to_string(mut_tag) + comma + to_string(static_pointer_cast<tree_evaluator>(evals.back())->gamma);
}

evaluator_ptr team_evaluator::clone() const {
  team_evaluator::ptr child(new team_evaluator(*this));
  child->evals = map<evaluator_ptr, evaluator_ptr>([](evaluator_ptr e) { return e->clone(); }, evals);
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