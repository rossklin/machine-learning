#include "tree_evaluator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "utility.hpp"

#define VERBOSE false

using namespace std;

// tree evaluator
const int max_weighted_subtrees = 10;

tree_evaluator::tree_evaluator(int d) : depth(d), evaluator() {
  gamma = fabs(rnorm(0.001, 0.0005));
  tag = "tree";
}

tree_evaluator::tree_evaluator() : evaluator() {
  gamma = fabs(rnorm(0.001, 0.0005));
  tag = "tree";
}

const hm<string, t_unary> &tree_evaluator::unary_ops() {
  static hm<string, t_unary> unary_op;

  MutexType m;
  m.Lock();
  if (unary_op.empty()) {
    unary_op["sin"].f = [](double x) { return sin(x); };
    unary_op["sin"].fprime = [](double x) { return cos(x); };
    unary_op["cos"].f = [](double x) { return cos(x); };
    unary_op["cos"].fprime = [](double x) { return -sin(x); };
    unary_op["atan"].f = [](double x) { return atan(x); };
    unary_op["atan"].fprime = [](double x) { return 1 / (1 + pow(x, 2)); };
    unary_op["sigmoid"].f = [](double x) { return 1 / (1 + exp(-x)); };
    unary_op["sigmoid"].fprime = [](double x) -> double {
      if (fabs(x) > 20) {
        return 0;
      } else {
        return exp(-x) / pow(1 + exp(-x), 2);
      }
    };

    unary_op["abs"].f = [](double x) { return fabs(x); };
    unary_op["abs"].fprime = [](double x) { return (x > 0) - (x < 0); };
  }
  m.Unlock();

  return unary_op;
}

const hm<string, t_binary> &tree_evaluator::binary_ops() {
  static hm<string, t_binary> binary_op;

  MutexType m;
  m.Lock();
  if (binary_op.empty()) {
    binary_op["kernel"].f = [](double x, double h) -> double {
      if (h > 0 && pow(x / h, 2) < 40) {
        return exp(-pow(x / h, 2));
      } else {
        return 0;
      }
    };

    binary_op["kernel"].dfdx1 = [](double x, double h) -> double {
      if (h > 0 && pow(x / h, 2) < 40) {
        return -2 * x / pow(h, 2) * exp(-pow(x / h, 2));
      } else {
        return 0;
      }
    };

    binary_op["kernel"].dfdx2 = [](double x, double h) -> double {
      if (h > 0 && pow(x / h, 2) < 40) {
        return 2 * pow(x, 2) / pow(h, 3) * exp(-pow(x / h, 2));
      } else {
        return 0;
      }
    };

    binary_op["product"].f = [](double a, double b) { return a * b; };
    binary_op["product"].dfdx1 = [](double a, double b) { return b; };
    binary_op["product"].dfdx2 = [](double a, double b) { return a; };

    // binary_op["ratio"].f = [](double a, double b) {
    //   if (fabs(b) > 0) {
    //     double q = a / b;
    //     return fmin(fabs(q), 1e12) * signum(q);
    //   } else {
    //     return 1e12 * signum(a);
    //   }
    // };

    // binary_op["ratio"].dfdx1 = [](double a, double b) -> double {
    //   if (fabs(b) > 0) {
    //     double q = a / b;
    //     if (fabs(q) < 1e12) {
    //       return 1 / b;
    //     } else {
    //       return 0;
    //     }
    //   } else {
    //     return 0;
    //   }
    // };

    // binary_op["ratio"].dfdx2 = [](double a, double b) -> double {
    //   if (fabs(b) > 0) {
    //     double q = a / b;
    //     if (fabs(q) < 1e12) {
    //       return -a / pow(b, 2);
    //     } else {
    //       return 0;
    //     }
    //   } else {
    //     return 0;
    //   }
    // };
  }
  m.Unlock();

  return binary_op;
}

double tree_evaluator::complexity() const { return root->count_trees(); }

double tree_evaluator::complexity_penalty() const {
  int n = root->count_trees();
  return 1 - kernel(n, 1000);
}

// todo
void tree_evaluator::tree::deserialize(stringstream &ss) {
  // read a start token
  string test;

  ss >> test;

  if (test != "{") {
    throw runtime_error("tree::deserialize: invalid start token: " + test);
  }

  int cid;
  int numsub;
  ss >> w >> cid;
  class_id = (tree_class)cid;

  if (class_id == CONSTANT_TREE) {
    ss >> const_value;
  } else if (class_id == INPUT_TREE) {
    ss >> input_index;
  } else if (class_id == WEIGHT_TREE) {
    ss >> numsub;
    assert(numsub > 0 && numsub <= max_weighted_subtrees);
  } else {
    ss >> fname;
  }

  // create subtrees
  if (class_id == UNARY_TREE) {
    subtree.resize(1);
  } else if (class_id == BINARY_TREE) {
    subtree.resize(2);
  } else if (class_id == WEIGHT_TREE) {
    subtree.resize(numsub);
  } else {
    subtree.clear();
  }

  for (auto &t : subtree) {
    t = tree::ptr(new tree);
    t->deserialize(ss);
  }

  ss >> test;
  if (test != "}") {
    throw runtime_error("tree::deserialize: invalid end token: " + test);
  }
}

string tree_evaluator::tree::serialize() const {
  stringstream ss;
  string sep = " ";

  ss << "{" << sep << w << sep << class_id << sep;

  if (class_id == CONSTANT_TREE) {
    ss << const_value;
  } else if (class_id == INPUT_TREE) {
    ss << input_index;
  } else if (class_id == WEIGHT_TREE) {
    ss << subtree.size();
  } else {
    ss << fname;
  }

  ss << sep;
  for (auto t : subtree) ss << sep << t->serialize() << sep;
  ss << "}";

  return ss.str();
}

// mutate in place
void tree_evaluator::tree::mutate(int dim) {
  if (u01() < 0.1) {
    w += rnorm(0, 0.1);
  }

  double p_grow = 0.01;
  double p_reduce = 0.01;

  if (subtree.size() > 0) {
    if (u01() < p_reduce) {
      // drop subtrees and become const/input
      subtree.clear();

      if (u01() < 0.5) {
        class_id = CONSTANT_TREE;
        const_value = resbuf / w + rnorm(0, 0.1);
      } else {
        class_id = INPUT_TREE;
        input_index = rand_int(0, dim - 1);
      }
    } else {
      for (auto t : subtree) t->mutate(dim);
    }
  } else {
    if (u01() < p_grow) {
      // extend tree
      if (u01() < 0.33) {
        // unary
        class_id = WEIGHT_TREE;
        subtree.resize(rand_int(2, max_weighted_subtrees));
      } else if (u01() < 0.5) {
        // unary
        class_id = UNARY_TREE;
        fname = sample_one(hm_keys(unary_ops()));
        subtree.resize(1);
      } else {
        // binary
        class_id = BINARY_TREE;
        fname = sample_one(hm_keys(binary_ops()));
        subtree.resize(2);
      }

      for (auto &t : subtree) {
        int n = min(dim, rand_int(2, 20));
        vector<int> ibuf = vector_sample(seq(0, dim - 1), n);
        t = tree::ptr(new tree);
        t->initialize(ibuf);
      }
    } else if (class_id == CONSTANT_TREE && u01() < 0.1) {
      // modify constant
      const_value += rnorm(0, 0.1);
    } else if (class_id == INPUT_TREE && u01() < 0.1) {
      // process different index
      input_index = rand_int(0, dim - 1);
    }
  }
}

int tree_evaluator::tree::count_trees() {
  int sum = 1;
  for (auto t : subtree) sum += t->count_trees();
  return sum;
}

bool tree_evaluator::tree::descendant_exists(tree *a, int lev) {
  assert(lev < 1000);
  if (this == a) return true;
  for (auto t : subtree)
    if (t->descendant_exists(a, lev + 1)) return true;
  return false;
}

bool tree_evaluator::tree::loop_free(int lev) {
  assert(lev < 1000);

  for (auto t : subtree) {
    if (t->descendant_exists(this) || !t->loop_free(lev + 1)) return false;
  }

  return true;
}

void tree_evaluator::tree::prune(double l) {
  if (fabs(w) <= l || !isfinite(w)) {
    subtree.clear();
    class_id = CONSTANT_TREE;
    const_value = 0;
    w = 1;
  }

  for (auto t : subtree) t->prune(l);
}

tree_evaluator::tree::ptr tree_evaluator::tree::clone() {
  tree::ptr t(new tree(*this));
  for (auto &s : t->subtree) s = s->clone();
  return t;
}

double tree_evaluator::tree::evaluate(const vec &x) {
  double val;

  if (class_id == CONSTANT_TREE) {
    val = const_value;
  } else if (class_id == INPUT_TREE) {
    val = x[input_index];
  } else if (class_id == UNARY_TREE) {
    val = unary_ops().at(fname).f(subtree[0]->evaluate(x));
  } else if (class_id == BINARY_TREE) {
    val = binary_ops().at(fname).f(subtree[0]->evaluate(x), subtree[1]->evaluate(x));
  } else if (class_id == WEIGHT_TREE) {
    val = 0;
    for (auto a : subtree) val += a->evaluate(x);
  } else {
    throw runtime_error("Invalid tree class id!");
  }

  return resbuf = w * val;
}

void tree_evaluator::tree::initialize(vector<int> inputs) {
  subtree.clear();
  w = rnorm();

  if (inputs.size() > 1) {
    if (u01() < 0.3) {
      // weight
      class_id = WEIGHT_TREE;
      int size = ranked_sample(seq(2, max_weighted_subtrees), 0.5);
      subtree.resize(size);
    } else if (u01() < 0.5) {
      // unary
      class_id = UNARY_TREE;
      fname = sample_one(hm_keys(unary_ops()));
      subtree.resize(1);
    } else {
      // binary
      class_id = BINARY_TREE;
      fname = sample_one(hm_keys(binary_ops()));
      subtree.resize(2);
    }
  } else if (inputs.size() == 1) {
    // input parameter
    class_id = INPUT_TREE;
    input_index = inputs.front();
  } else {
    // constant
    class_id = CONSTANT_TREE;
    const_value = rnorm();
  }

  for (int i = 0; i < subtree.size(); i++) {
    float m = inputs.size() / float(subtree.size() - i);
    int ntake = rnorm(m, m / 2);
    if (ntake < 0) ntake = 0;
    if (ntake > inputs.size() || i == subtree.size() - 1) ntake = inputs.size();

    vector<int> ibuf;
    if (ntake > 0) {
      ibuf.insert(ibuf.begin(), inputs.begin(), inputs.begin() + ntake);
      inputs.erase(inputs.begin(), inputs.begin() + ntake);
    }

    subtree[i] = tree::ptr(new tree);
    subtree[i]->initialize(ibuf);
  }
}

tree_evaluator::tree::ptr tree_evaluator::tree::get_subtree(double p_cut) {
  if (subtree.empty()) return shared_from_this();

  if (u01() < p_cut) {
    return sample_one(subtree)->clone();
  } else {
    return sample_one(subtree)->get_subtree(p_cut);
  }
}

void tree_evaluator::tree::emplace_subtree(tree::ptr x, double p_put) {
  if (subtree.empty()) {
    // become a weight tree with x as subtree
    class_id = WEIGHT_TREE;
    w = 1;
    subtree = {x};
  } else {
    if (u01() < p_put) {
      int idx = rand_int(0, subtree.size() - 1);
      subtree[idx] = x;
    } else {
      sample_one(subtree)->emplace_subtree(x, p_put);
    }
  }
}

// void tree_evaluator::tree::apply_dw(double scale) {
// #if VERBOSE
//   // debug
//   cout << "Node class " << class_id << ": apply dw: resbuf = " << resbuf << ", dwbuf = " << dwbuf << " :: w from " << w << " to " << (w - dwbuf * scale) << endl;
// #endif

//   double delta = -dwbuf * scale;

//   if (fabs(delta) >= fabs(w) && signum(delta) != signum(w)) {
//     w = 0;  //w was reduced to 0, stop here!
//   } else {
//     w += delta;  // basic gradient descent
//   }

//   ssw = pow(w, 2);
//   for (auto t : subtree) {
//     t->apply_dw(scale);
//     ssw += t->ssw;
//   }
// }

// void tree_evaluator::tree::scale_weights(double rescale) {
//   w *= rescale;
//   // ssw = pow(w, 2);
//   assert(isfinite(w));
//   for (auto t : subtree) {
//     t->scale_weights(rescale);
//     // ssw += t->ssw;
//   }
// }

// must run evaluate first to set resbuf
int tree_evaluator::tree::calculate_dw(vec &dgdw, int offset, double delta, double alpha, double gamma) {
  double dydw;
  if (w == 0) {
    dydw = 0;  // once a weight hits zero, leave it there for later pruning
  } else {
    dydw = alpha * resbuf / w;
  }
  dgdw[offset] = -2 * delta * dydw - gamma * signum(w);
  offset++;

  if (class_id == BINARY_TREE) {
    double y1 = subtree[0]->resbuf;
    double y2 = subtree[1]->resbuf;
    double left_deriv = binary_ops().at(fname).dfdx1(y1, y2);
    double right_deriv = binary_ops().at(fname).dfdx2(y1, y2);
    offset = subtree[0]->calculate_dw(dgdw, offset, delta, w * left_deriv * alpha, gamma);
    offset = subtree[1]->calculate_dw(dgdw, offset, delta, w * left_deriv * alpha, gamma);
  } else if (class_id == UNARY_TREE) {
    double y = subtree[0]->resbuf;
    double deriv = unary_ops().at(fname).fprime(y);
    offset = subtree[0]->calculate_dw(dgdw, offset, delta, w * deriv * alpha, gamma);
  } else if (class_id == WEIGHT_TREE) {
    for (auto a : subtree) {
      offset = a->calculate_dw(dgdw, offset, delta, w * alpha, gamma);
    }
  }

  return offset;
}

set<int> tree_evaluator::tree::list_inputs() const {
  if (class_id == INPUT_TREE) {
    return {input_index};
  } else if (subtree.size() > 0) {
    set<int> res;
    for (auto t : subtree) res = set_union(res, t->list_inputs());
    return res;
  } else {
    return {};
  }
}

void tree_evaluator::tree::add_inputs(vector<int> inputs) {
  if (inputs.empty()) return;

  bool do_init = false;
  if (inputs.size() > 1) {
    if (subtree.empty()) {
      class_id = WEIGHT_TREE;
      subtree.resize(ranked_sample(seq(2, max_weighted_subtrees), 0.5));
      for (auto &t : subtree) t = ptr(new tree);
      do_init = true;  // subtrees need to be initialized
    }
  } else if (subtree.empty()) {
    class_id = INPUT_TREE;
    input_index = inputs.front();
  }

  if (subtree.size()) {
    vector<vector<int>> parts = random_partition<int>(inputs, subtree.size());
    for (int i = 0; i < subtree.size(); i++) {
      tree::ptr t = subtree[i];
      if (do_init) {
        t->initialize(parts[i]);
      } else {
        t->add_inputs(parts[i]);
      }
    }
  }
}

int tree_evaluator::tree::set_weights(const vec &x, int offset) {
  assert(offset < x.size());

  w = x[offset];
  offset++;

  for (auto a : subtree) offset = a->set_weights(x, offset);

  return offset;
}

vec tree_evaluator::tree::get_weights() const {
  vec res = {w};
  for (auto a : subtree) res = vec_append(res, a->get_weights());
  return res;
}

evaluator_ptr tree_evaluator::clone() const {
  shared_ptr<tree_evaluator> e(new tree_evaluator(*this));
  e->root = root->clone();
  return e;
}

double tree_evaluator::evaluate(vec x) {
  return root->evaluate(x);
}

void tree_evaluator::prune(double l) {
  root->prune(l);
}

bool tree_evaluator::update(vector<record> results, int age, double &rel_change) {
#if VERBOSE
  // debug
  vec printbuf(input.begin(), input.begin() + 5);
  cout << "Update: input = [" << printbuf << "]" << endl;
#endif

  rel_change = 0;
  // double current = evaluate(input);
  // double delta = output - current;

  // // compute weight increments
  // root->calculate_dw(delta, 1, gamma, stable);

  // double wlen = sqrt(root->ssw);
  // double dwlen = sqrt(root->ssdw);
  // if (dwlen == 0) {
  //   return false;
  // }

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
    tree::ptr buf = root->clone();
    buf->set_weights(x);
    for (auto res : results) {
      double test = buf->evaluate(res.input);
      G += pow(test - res.sum_future_rewards, 2);
    }
    return G;
  };

  // calculate dG/dw for whole batch
  auto fgrad = [this, results](vec x) -> vec {
    int n = x.size();
    vec dgdw(n, 0);
    tree::ptr buf = root->clone();
    buf->set_weights(x);

    for (auto res : results) {
      double output = buf->evaluate(res.input);
      double delta = res.sum_future_rewards - output;
      vec grad(x);
      buf->calculate_dw(grad, 0, delta, 1, gamma);
      dgdw = dgdw + grad;
    }

    return dgdw;
  };

  vec x0 = root->get_weights();
  vec x = voptim(x0, fopt, fgrad);

  if (fopt(x) >= fopt(x0)) {
    rel_change = INFINITY;
    return false;
  }

  // double step = foptim(1e-4 * wlen / dwlen, fopt);
  // double h = fopt(step);

  // // debug: print opt landscape
  // double ss = step / 100;
  // dbgbuf.resize(300);
  // for (int i = 0; i < 300; i++) dbgbuf[i] = to_string(fopt(i * ss));
  // cout << "step <- " << ss << endl;
  // cout << "landscape <- c(" << join_string(dbgbuf, ",\n") << ")" << endl;

  // cout << "RESULT: " << h / pow(delta, 2) << endl;

  // check if optimization succeeded
  // if (h >= pow(delta, 2) || step < 0) {
  //   rel_change = INFINITY;
  //   return false;
  // }

  double time_scale = 1 / sqrt(age + 1);
  double limit = 0.1 * time_scale;  // don't allow stepping more than .1% of weight vector length

  vec step = learning_rate * time_scale * (x - x0);

  if (l2norm(step) > limit * l2norm(x0)) {
    step = limit * l2norm(x0) / l2norm(x - x0) * (x - x0);
  }

  rel_change = l2norm(step) / l2norm(x0);

  root->set_weights(x0 + step);

  // root->apply_dw(step);

  // remove subtrees where the weight was reduced to zero
  prune();

  // scale down weights if too large
  x = root->get_weights();
  double wlen = l2norm(x);
  stable = isfinite(wlen);

  if (stable && wlen > weight_limit) {
    root->set_weights(weight_limit / wlen * x);
    prune();
  }

  return stable;
}

evaluator_ptr tree_evaluator::mate(evaluator_ptr partner_buf) const {
  shared_ptr<tree_evaluator> partner = static_pointer_cast<tree_evaluator>(partner_buf);
  shared_ptr<tree_evaluator> child = static_pointer_cast<tree_evaluator>(clone());
  child->learning_rate = fmax(rnorm(0.5, 0.1) * (learning_rate + partner->learning_rate), 1e-5);
  child->weight_limit = fmax(rnorm(0.5, 0.1) * (weight_limit + partner->weight_limit), 1);

  tree::ptr sub = partner->root->get_subtree(0.3);
  child->root->emplace_subtree(sub, 0.3);
  return child->mutate();
}

evaluator_ptr tree_evaluator::mutate() const {
  shared_ptr<tree_evaluator> child = static_pointer_cast<tree_evaluator>(clone());
  child->root->mutate(dim);
  if (u01() < 0.1) child->learning_rate = fmax(child->learning_rate + rnorm(0, 0.1), 1e-5);
  if (u01() < 0.1) child->weight_limit = fmax(child->weight_limit * rnorm(1, 0.1), 1);
  return child;
}

string tree_evaluator::serialize() const {
  stringstream ss;
  string sep = " ";
  ss << evaluator::serialize() << sep << learning_rate << sep << weight_limit << sep << gamma << sep << root->serialize();
  return ss.str();
}

void tree_evaluator::deserialize(stringstream &ss) {
  evaluator::deserialize(ss);
  ss >> learning_rate >> weight_limit >> gamma;

  root = tree::ptr(new tree);
  root->deserialize(ss);

  return;
}

void tree_evaluator::set_learning_rate(double r) { learning_rate = r; }

void tree_evaluator::initialize(input_sampler sampler, int cdim, set<int> ireq) {
  stable = true;
  dim = cdim;
  learning_rate = fabs(rnorm(0, 0.5));
  weight_limit = u01(100, 10000);

  // prepare input indices for the tree to use
  vector<int> ibuf;
  for (int i = 0; i < cdim; i++) {
    int n = ranked_sample(seq(0, 3), 0.5);
    if (ireq.count(i)) n = ranked_sample(seq(1, 3), 0.8);
    ibuf.insert(ibuf.end(), n, i);
  }
  random_shuffle(ibuf.begin(), ibuf.end());

  root = tree::ptr(new tree);
  root->initialize(ibuf);
}

string tree_evaluator::status_report() const {
  stringstream ss;
  string sep = ",";
  ss << root->count_trees() << sep << learning_rate;
  return ss.str();
}

set<int> tree_evaluator::list_inputs() const {
  return root->list_inputs();
}