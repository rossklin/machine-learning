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

bool evaluator::mem_update(vector<record> results, agent_ptr a, double &rel_change) {
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

  // // find minimum of discrepancy wrt step size
  // auto fopt = [this, results](vec x) -> double {
  //   double G = 0;
  //   evaluator_ptr buf = clone();
  //   buf->set_weights(x);
  //   for (auto res : results) {
  //     double test = buf->evaluate(res.input);
  //     G += pow(test - res.sum_future_rewards, 2);
  //   }
  //   return G;
  // };

  // // calculate dG/dw for whole batch
  // auto fgrad = [this, results, a](vec x) -> vec {
  //   int n = x.size();
  //   vec dgdw(n, 0);
  //   evaluator_ptr buf = clone();
  //   buf->set_weights(x);

  //   for (auto res : results) {
  //     // double output = buf->evaluate(res.input);
  //     // double delta = res.sum_future_rewards - output;
  //     // vec grad(x);
  //     // buf->calculate_dw(grad, 0, delta, 1, gamma);
  //     dgdw = dgdw + buf->gradient(res.input, res.sum_future_rewards, a->w_reg);
  //   }

  //   return dgdw;
  // };

  // vec xprev = get_weights();

  // int restarts = 5;
  // int wdim = xprev.size();
  // typedef pair<double, vec> scored_vec;
  // double y0 = fopt(xprev);
  // vector<scored_vec> xs = vec_replicate<scored_vec>(
  //     [fopt, fgrad, wdim, xprev, a, y0]() {
  //       vec x0 = xprev + vec_replicate<double>(bind(rnorm, 0, 1), wdim);
  //       optim_result<double> res = voptim(x0, fopt, fgrad);

  //       // update agent optim_stats
  //       a->optim_stats.its.push(res.its);
  //       a->optim_stats.overshoot.push(res.overshoot);
  //       a->optim_stats.success.push(res.success);
  //       a->optim_stats.dx.push(res.dx);
  //       a->optim_stats.dy.push(res.dy);

  //       if (res.success) {
  //         return make_pair(res.obj, res.opt);
  //       } else {
  //         return make_pair(fopt(x0), x0);
  //       }
  //     },
  //     restarts);

  // sort(xs.begin(), xs.end(), [](scored_vec a, scored_vec b) { return a.first < b.first; });
  // vec x = xs.front().second;

  // a->optim_stats.improvement.push((y0 - xs.front().first) / y0);  // Improvement negative if optimum is worse than fopt(xprev)

  // if (fopt(x) >= fopt(xprev)) {
  //   rel_change = INFINITY;

  //   // // debug
  //   // cout << "update: failed" << endl;

  //   return false;
  // }

  // double perf_w = mem_weight(a->score_simple.value_ma, a->score_refbot.value_ma);  // weight memory by current performance
  // memories.insert(memories.begin(), make_pair(n * perf_w, x));

  // // todo: cluster and weight memories

  // if (memories.size() > a->mem_limit && a->mem_limit > 0) {
  //   // merge extra memories
  //   int i = a->mem_limit - 1;
  //   memories[i] = make_pair(memories[i].first + memories[i + 1].first, 0.5 * (memories[i].second + memories[i + 1].second));
  //   memories.resize(a->mem_limit);
  // }

  // // weight memories by number of data included and memory curve
  // vec w = map<int, double>(
  //     [this, a](int i) -> double {
  //       return memories[i].first * (2 + sigmoid(a->mem_limit - a->mem_curve - i, a->mem_curve / 3));
  //     },
  //     seq(0, memories.size() - 1));

  // double wsum = sum(w);

  // // update weights using weighted sum of memories
  // x = vec(x.size(), 0);
  // for (int i = 0; i < memories.size(); i++) x = x + w[i] / wsum * memories[i].second;
  // set_weights(x);

  // rel_change = l2norm(x - xprev) / l2norm(xprev);
  // stable = stable && isfinite(rel_change);

  // // if (a->age > 10 && rel_change > 0.2) {
  // //   // debug
  // //   cout << "-------------------------------------------" << endl;
  // //   cout << "Add mem with w " << n << "*" << perf_w << "=" << memories.front().first << endl;
  // //   cout << "perf_w: " << perf_w << endl;
  // //   cout << "Mem curve: " << (2 + sigmoid(a->mem_limit - a->mem_curve, a->mem_curve / 3)) << ", " << (2 + sigmoid(a->mem_limit - a->mem_curve - memories.size() + 1, a->mem_curve / 3)) << endl;
  // //   cout << "Memory weight vector: " << w << endl;
  // //   cout << "Rel change: " << rel_change << endl;
  // //   cout << "-------------------------------------------" << endl;
  // // }

  return stable;
}