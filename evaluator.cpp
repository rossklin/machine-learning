#include "evaluator.hpp"

#include <sstream>

#include "agent.hpp"
#include "team_evaluator.hpp"
#include "tree_evaluator.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

evaluator::evaluator() {
  mut_tag = (dist_category)-1;
  learning_rate = -1;
  tag = "evaluator";
  stable = true;
}

string evaluator::serialize() const {
  stringstream ss;
  ss << dim << sep << stable << sep << learning_rate << sep << mut_tag;
  return ss.str();
}

void evaluator::deserialize(stringstream &ss) {
  int buf;
  ss >> dim >> stable >> learning_rate >> buf;
  mut_tag = (dist_category)buf;
}

void evaluator::set_learning_rate(double r) { learning_rate = r; }

evaluator_ptr deserialize_evaluator(stringstream &ss) {
  evaluator_ptr eval;
  string tag;
  ss >> tag;

  if (tag == "tree") {
    eval = evaluator_ptr(new tree_evaluator);
  } else if (tag == "team") {
    eval = evaluator_ptr(new team_evaluator);
  } else {
    throw runtime_error("Invalid evaluator tag: " + tag);
  }

  eval->deserialize(ss);
  return eval;
}

string serialize_evaluator(evaluator_ptr e) {
  stringstream ss;
  ss << e->tag << sep << e->serialize();
  return ss.str();
}

bool evaluator::update(vector<record> results, agent_ptr a, double &rel_change) {
  // move to agent conf param
  rel_change = 0;
  int n = results.size();

  // Single data point:
  // G = d² = (T-Y)²
  // dG/dwi = -2 d dy/dw

  // Multiple data point:
  // d[i] = T[i] - Y[i]
  // G = ||d||² = Sum { (T[i] - Y[i])²}
  // dG[i] / dw[j] = -2 d[i] dy[i]/dw[j]
  // dG/dw[j] = Sum_i dG[i]/dw[j]

  // find minimum of discrepancy wrt step size
  auto fopt = [this, results](vec x) -> double {
    double G = 0;
    evaluator_ptr buf = clone();
    buf->set_weights(x);
    for (auto res : results) {
      double test = buf->evaluate(res.input);
      G += pow(test - res.sum_future_rewards, 2);
    }
    return G;
  };

  // calculate dG/dw for whole batch
  auto fgrad = [this, results, a](vec x) -> vec {
    int n = x.size();
    vec dgdw(n, 0);
    evaluator_ptr buf = clone();
    buf->set_weights(x);

    for (auto res : results) {
      // double output = buf->evaluate(res.input);
      // double delta = res.sum_future_rewards - output;
      // vec grad(x);
      // buf->calculate_dw(grad, 0, delta, 1, gamma);
      dgdw = dgdw + buf->gradient(res.input, res.sum_future_rewards, a->w_reg);
    }

    return dgdw;
  };

  // todo: random restarts?
  vec x0 = get_weights();
  vec x = voptim(x0, fopt, fgrad);

  if (fopt(x) >= fopt(x0)) {
    rel_change = INFINITY;
    return false;
  }

  memories.insert(memories.begin(), make_pair(n, x));

  // todo: cluster and weight memories

  if (memories.size() > a->mem_limit) {
    // merge extra memories
    int i = a->mem_limit - 1;
    memories[i] = make_pair(memories[i].first + memories[i + 1].first, 0.5 * (memories[i].second + memories[i + 1].second));
    memories.resize(a->mem_limit);
  }

  vec w = map<int, double>(
      [this, a](int i) -> double {
        return memories[i].first * (1 + sigmoid(a->mem_limit - a->mem_curve - i, a->mem_curve / 3));
      },
      seq(0, memories.size() - 1));

  double wsum = sum(w);

  x = vec(x.size(), 0);
  for (int i = 0; i < memories.size(); i++) x = x + w[i] / wsum * memories[i].second;
  set_weights(x);

  rel_change = l2norm(x - x0) / l2norm(x0);
  stable = stable && isfinite(rel_change);

  return stable;
}