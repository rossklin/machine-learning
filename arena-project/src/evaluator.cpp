#include "evaluator.hpp"

#include <algorithm>
#include <nlopt.hpp>
#include <sstream>

#include "agent.hpp"
#include "team_evaluator.hpp"
#include "tree_evaluator.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

evaluator::evaluator() {
  mut_tag = (dist_category)-1;
  tag = "evaluator";
  stable = true;
}

string evaluator::serialize() const {
  stringstream ss;
  ss << dim << sep << stable << sep << mut_tag;
  return ss.str();
}

void evaluator::deserialize(stringstream &ss) {
  int buf;
  ss >> dim >> stable >> buf;
  mut_tag = (dist_category)buf;
}

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

void evaluator::reset_memory_weights(double a) {
  for (auto &m : memories) m.first *= a;
}

evaluator_ptr evaluator::update(vector<record> results, agent_ptr a, double &rel_change) const {
  auto buf = clone();
  if (buf->mod_update(results, a, rel_change).success) {
    return buf;
  } else {
    return NULL;
  }
}

double f0c(double y, double d) {
  if (d > 0) {
    d = min(d, y / d);
  } else {
    d = max(d, y / d);
  }
  return d;
}

typedef function<double(const std::vector<double> &, std::vector<double> &)> ftype;
double nlopt_f(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  ftype *f = reinterpret_cast<ftype *>(my_func_data);
  return (*f)(x, grad);
}

// TODO: seems a solution that returns constant 0 is basically always an attractive local optimum
// TODO: test with simple constructed tree with known solution and gradient
optim_result<double> evaluator::run_nlopt(vector<record> results, agent_ptr a, double &rel_change) {
  // move to agent conf param
  rel_change = 0;
  int n = results.size();

  vec x = get_weights();
  int dim = x.size();

  evaluator_ptr buf = clone();

  auto fopt = [this, results, a, buf](const std::vector<double> &x) -> double {
    buf->set_weights(x);
    // Compute opt value
    double G = 0;
    for (auto res : results) {
      res.opts[res.selected_option].output = (1 - a->learning_rate) * res.opts[res.selected_option].output + a->learning_rate * res.sum_future_rewards;
      // G = sum(Gi), Gi = (Ti - Yi)²
      for (int i = 0; i < res.opts.size(); i++) {
        double test = buf->evaluate(res.opts[i].input);
        G += pow(test - res.opts[i].output, 2);
      }
    }

    // regularization component
    for (auto w : x) G += a->w_reg * fabs(w);
    return G;
  };

  ftype f = [this, results, a, buf, fopt](const std::vector<double> &x, std::vector<double> &grad) -> double {
    if (x.size() != grad.size()) {
      throw logic_error("Bad gradient dim");
    }

    buf->set_weights(x);
    // Compute gradient
    int n = x.size();
    vec dgdw(n, 0);

    for (auto res : results) {
      res.opts[res.selected_option].output = (1 - a->learning_rate) * res.opts[res.selected_option].output + a->learning_rate * res.sum_future_rewards;
      // dG/dwj = sum(dGi/dwj)
      for (int i = 0; i < res.opts.size(); i++) {
        dgdw = dgdw + buf->gradient(res.opts[i].input, res.opts[i].output);
      }
    }

    // regularization component
    for (int i = 0; i < n; i++) grad[i] = dgdw[i] + a->w_reg * signum(x[i]);

    // scale down grad so nlopt will chill a bit
    double gn = l2norm(grad);
    if (gn > 0.01 * n) grad = 0.01 * n / gn * grad;

    // Return opt value
    double y = fopt(x);
    cout << "New objective: " << y << " at " << x << endl;
    cout << " -- gradient " << grad << endl;
    return y;
  };

  // NLOPT code here
  nlopt::opt opt(nlopt::LD_LBFGS, dim);
  opt.set_min_objective(nlopt_f, &f);
  opt.set_maxeval(20);
  opt.set_lower_bounds(vec(dim, -10));
  opt.set_upper_bounds(vec(dim, 10));
  double minf;
  double y = fopt(x);
  vec x0 = x;

  optim_result<double> res;
  try {
    nlopt::result result = opt.optimize(x, minf);
    cout << "found minimum " << minf << endl;
    set_weights(x);
    res.success = true;
    res.obj = minf;
    res.improvement = (y - minf) / y;
    cout << "change x: " << l2norm(x - x0) << endl;
    cout << "change y: from " << y << " to " << minf << endl;
  } catch (std::exception &e) {
    cout << "nlopt failed: " << e.what() << endl;
    res.success = false;
    res.obj = y;
    res.improvement = 0;
  }

  return res;
}

vec evaluator::fgrad(vec x, vector<record> results, agent_ptr a) {
  int n = x.size();
  vec dgdw(n, 0);
  evaluator_ptr buf = clone();
  buf->set_weights(x);

  for (auto res : results) {
    res.opts[res.selected_option].output = (1 - a->learning_rate) * res.opts[res.selected_option].output + a->learning_rate * res.sum_future_rewards;
    // dG/dwj = sum(dGi/dwj)
    for (int i = 0; i < res.opts.size(); i++) {
      dgdw = dgdw + buf->gradient(res.opts[i].input, res.opts[i].output);
    }
  }

  // regularization component
  for (int i = 0; i < n; i++) dgdw[i] += a->w_reg * signum(x[i]);

  return dgdw;
}

optim_result<double> evaluator::mod_update(vector<record> results, agent_ptr a, double &rel_change) {
  // move to agent conf param
  rel_change = 0;
  int n = results.size();

  auto fopt = [this, results, a](vec x) -> double {
    double G = 0;
    evaluator_ptr buf = clone();
    buf->set_weights(x);
    for (auto res : results) {
      res.opts[res.selected_option].output = (1 - a->learning_rate) * res.opts[res.selected_option].output + a->learning_rate * res.sum_future_rewards;
      // G = sum(Gi), Gi = (Ti - Yi)²
      for (int i = 0; i < res.opts.size(); i++) {
        double test = buf->evaluate(res.opts[i].input);
        G += pow(test - res.opts[i].output, 2);
      }
    }

    // regularization component
    for (auto w : x) G += a->w_reg * fabs(w);

    return G;
  };

  vec x = get_weights();
  double y = fopt(x);
  vec g = fgrad(x, results, a);
  if (a->use_f0c) g = map<double, double>(bind(f0c, y, placeholders::_1), g);
  vec delta = -1 * g;
  rel_change = l2norm(delta) / l2norm(x);

  if (rel_change > a->step_limit) {
    delta = a->step_limit / rel_change * delta;
    rel_change = a->step_limit;
  }

  double y2 = fopt(x + delta);
  stable = stable && isfinite(y2);

  optim_result<double> res;

  res.success = y2 < y;
  res.improvement = (y - y2) / y;
  res.obj = y2;

  if (res.success) {
    set_weights(x + delta);
  }

  cout << "New objective " << y2 << " at " << (x + delta) << endl;
  cout << " -- gradient: " << g << endl;

  return res;
}
