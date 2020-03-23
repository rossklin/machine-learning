#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

#include "tree_evaluator.hpp"
#include "utility.hpp"

#define VERBOSE false

using namespace std;

// tree evaluator
const int max_weighted_subtrees = 10;

tree_evaluator::tree_evaluator(int d) : depth(d), evaluator() {
  gamma = abs(rnorm(0.001, 0.0005));
}

hm<string, t_unary> tree_evaluator::unary_ops() {
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
      if (abs(x) > 20) {
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

hm<string, t_binary> tree_evaluator::binary_ops() {
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

    binary_op["ratio"].f = [](double a, double b) {
      if (fabs(b) > 0) {
        double q = a / b;
        return fmin(fabs(q), 1e12) * signum(q);
      } else {
        return 1e12 * signum(a);
      }
    };

    binary_op["ratio"].dfdx1 = [](double a, double b) -> double {
      if (fabs(b) > 0) {
        double q = a / b;
        if (fabs(q) < 1e12) {
          return 1 / b;
        } else {
          return 0;
        }
      } else {
        return 0;
      }
    };

    binary_op["ratio"].dfdx2 = [](double a, double b) -> double {
      if (fabs(b) > 0) {
        double q = a / b;
        if (fabs(q) < 1e12) {
          return -a / pow(b, 2);
        } else {
          return 0;
        }
      } else {
        return 0;
      }
    };
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
  double p_reduce = 0.001;

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
      if (u01() < 0.5) {
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
        t = tree::ptr(new tree);
        t->initialize(dim, rand_int(1, 5));
      }
    } else if (class_id == CONSTANT_TREE && u01() < 0.1) {
      // modify constant
      const_value += rnorm(0, 0.1);
    } else if (class_id == INPUT_TREE && u01() < 0.01) {
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

void tree_evaluator::tree::prune() {
  if (w == 0) {
    subtree.clear();
    class_id = CONSTANT_TREE;
    const_value = 0;
    w = 1;
  }

  for (auto t : subtree) t->prune();
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
    val = unary_ops()[fname].f(subtree[0]->evaluate(x));
  } else if (class_id == BINARY_TREE) {
    val = binary_ops()[fname].f(subtree[0]->evaluate(x), subtree[1]->evaluate(x));
  } else if (class_id == WEIGHT_TREE) {
    val = 0;
    for (auto a : subtree) val += a->evaluate(x);
  } else {
    throw runtime_error("Invalid tree class id!");
  }

  return resbuf = w * val;
}

void tree_evaluator::tree::initialize(int dim, int depth) {
  double p_cut = 1 / (double)(depth + 1);
  subtree.clear();
  w = rnorm();

  if (depth > 0 && u01() < (1 - p_cut)) {
    if (u01() < 0.4) {
      // weight
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
  } else if (u01() < 0.5) {
    // constant
    class_id = CONSTANT_TREE;
    const_value = rnorm();
  } else {
    // input parameter
    class_id = INPUT_TREE;
    input_index = ranked_sample(seq(0, dim - 1), 0.2);
  }

  for (auto &t : subtree) {
    t = tree::ptr(new tree);
    t->initialize(dim, depth - 1);
  }
}

tree_evaluator::tree::ptr tree_evaluator::tree::get_subtree(double p_cut) {
  if (subtree.empty()) return 0;

  if (u01() < p_cut) {
    return sample_one(subtree)->clone();
  } else {
    for (auto t : subtree) {
      tree::ptr
          test = t->get_subtree(p_cut);
      if (test) return test;
    }
    return get_subtree(p_cut);
  }
}

bool tree_evaluator::tree::emplace_subtree(tree::ptr x, double p_put) {
  if (subtree.empty()) return false;

  if (u01() < p_put) {
    int idx = rand_int(0, subtree.size() - 1);
    subtree[idx] = x;
    return true;
  } else {
    for (auto t : subtree) {
      if (t->emplace_subtree(x, p_put)) return true;
    }
    return emplace_subtree(x, p_put);
  }
}

void tree_evaluator::tree::apply_dw(double scale) {
#if VERBOSE
  // debug
  cout << "Node class " << class_id << ": apply dw: resbuf = " << resbuf << ", dwbuf = " << dwbuf << " :: w from " << w << " to " << (w - dwbuf * scale) << endl;
#endif

  double delta = -dwbuf * scale;

  if (abs(delta) >= abs(w) && signum(delta) != signum(w)) {
    w = 0;  //w was reduced to 0, stop here!
  } else {
    w += delta;  // basic gradient descent
  }

  ssw = pow(w, 2);
  for (auto t : subtree) {
    t->apply_dw(scale);
    ssw += t->ssw;
  }
}

void tree_evaluator::tree::scale_weights(double rescale) {
  w *= rescale;
  ssw = pow(w, 2);
  assert(isfinite(w));
  for (auto t : subtree) {
    t->scale_weights(rescale);
    ssw += t->ssw;
  }
}

void tree_evaluator::tree::calculate_dw(double delta, double alpha, double gamma, bool &stable) {
  ssw = 0;
  ssdw = 0;

  if (class_id == BINARY_TREE) {
    double y1 = subtree[0]->resbuf;
    double y2 = subtree[1]->resbuf;
    double left_deriv = binary_ops()[fname].dfdx1(y1, y2);
    double right_deriv = binary_ops()[fname].dfdx2(y1, y2);
    subtree[0]->calculate_dw(delta, w * left_deriv * alpha, gamma, stable);
    subtree[1]->calculate_dw(delta, w * right_deriv * alpha, gamma, stable);
  } else if (class_id == UNARY_TREE) {
    double y = subtree[0]->resbuf;
    double deriv = unary_ops()[fname].fprime(y);
    subtree[0]->calculate_dw(delta, w * deriv * alpha, gamma, stable);
  } else if (class_id == WEIGHT_TREE) {
    for (auto a : subtree) {
      a->calculate_dw(delta, w * alpha, gamma, stable);
    }
  }

  double dydw;
  if (w == 0) {
    dydw = 0;  // once a weight hits zero, leave it there for later pruning
  } else {
    dydw = alpha * resbuf / w;
  }

  for (auto t : subtree) {
    ssw += t->ssw;
    ssdw += t->ssdw;
  }

  dwbuf = -2 * delta * dydw - gamma * signum(w);  // dG/dw = -2 delta dy/dw, G = delta²
  ssw += pow(w, 2);
  ssdw += pow(dwbuf, 2);
}

evaluator_ptr tree_evaluator::clone() const {
  shared_ptr<tree_evaluator> e(new tree_evaluator(*this));
  e->root = root->clone();
  return e;
}

double tree_evaluator::evaluate(vec x) {
  return root->evaluate(x);
}

bool tree_evaluator::update(vec input, double output, int age) {
#if VERBOSE
  // debug
  vec printbuf(input.begin(), input.begin() + 5);
  cout << "Update: input = [" << printbuf << "]" << endl;
#endif

  double current = evaluate(input);
  double delta = output - current;

  // compute weight increments
  root->calculate_dw(delta, 1, gamma, stable);

  double wlen = sqrt(root->ssw);
  double dwlen = sqrt(root->ssdw);
  if (dwlen == 0) return false;

  double time_scale = 1 / (log(age + 1) + 1);
  double step = learning_rate * time_scale;  // preferred step is size of gradient
  double limit = 0.001 * time_scale;         // don't allow stepping more than 1% of weight vector length

  if (step * dwlen > limit * wlen) {
    step = limit * wlen / dwlen;
  }

  root->apply_dw(step);

  // remove subtrees where the weight was reduced to zero
  root->prune();

  // scale down weights if too large
  wlen = sqrt(root->ssw);
  double new_output = evaluate(input);
  stable = isfinite(new_output);

#if VERBOSE
  cout << "Update complete, output from " << current << " to " << new_output << " by " << (new_output - current) << ", target = " << output << endl;
#endif

  if (stable && wlen > weight_limit) {
#if VERBOSE
    cout << "Limiting weights" << endl;
#endif

    root->scale_weights(weight_limit / wlen);
    assert(sqrt(root->ssw) <= 1.1 * weight_limit);
  }

  return stable;
}

evaluator_ptr tree_evaluator::mate(evaluator_ptr partner_buf) const {
  shared_ptr<tree_evaluator> partner = static_pointer_cast<tree_evaluator>(partner_buf);
  shared_ptr<tree_evaluator> child = static_pointer_cast<tree_evaluator>(clone());
  child->learning_rate = fmax(rnorm(0.5, 0.1) * (learning_rate + partner->learning_rate), 1e-5);
  child->weight_limit = fmax(rnorm(0.5, 0.1) * (weight_limit + partner->weight_limit), 1);

  tree::ptr sub = 0;

  // todo: guarantee success
  for (int i = 0; i < 10 && sub == 0; i++) sub = partner->root->get_subtree(0.3);
  if (sub == 0) return 0;

  bool success = false;
  for (int i = 0; i < 10 && !success; i++) success = child->root->emplace_subtree(sub, 0.3);

  if (success) {
    // assert(child->root->loop_free());
    return child;
  } else {
    return 0;
  }
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
  ss << evaluator::serialize() << sep << learning_rate << sep << weight_limit << sep << root->serialize();
  return ss.str();
}

void tree_evaluator::deserialize(stringstream &ss) {
  evaluator::deserialize(ss);
  ss >> learning_rate >> weight_limit;

  root = tree::ptr(new tree);
  root->deserialize(ss);

  return;
}

void tree_evaluator::example_setup(int cdim) {
  stable = true;
  dim = cdim;
  learning_rate = fabs(rnorm(0, 0.01));
  weight_limit = u01(1, 10000);

  root = tree::ptr(new tree);

  // debug: fixed tree definition
  root->w = 1;
  root->class_id = BINARY_TREE;
  root->subtree.resize(2);
  root->fname = "product";

  tree::ptr K(new tree);
  K->w = 1;
  K->class_id = BINARY_TREE;
  K->fname = "kernel";
  K->subtree.resize(2);

  tree::ptr W(new tree);
  W->w = 1;
  W->class_id = WEIGHT_TREE;
  W->subtree.resize(2);

  tree::ptr SK1(new tree);
  tree::ptr SK2(new tree);

  SK2->class_id = UNARY_TREE;

  SK1->class_id = UNARY_TREE;
  SK1->w = 1;
  SK1->fname = "sigmoid";
  SK1->subtree.resize(1);

  SK2->class_id = UNARY_TREE;
  SK2->w = -1;
  SK2->fname = "sigmoid";
  SK2->subtree.resize(1);

  W->subtree[0] = SK1;
  W->subtree[1] = SK2;

  tree::ptr I1(new tree);
  I1->class_id = INPUT_TREE;
  I1->w = 1;
  I1->input_index = 0;

  SK1->subtree[0] = I1;

  tree::ptr I2(new tree);
  I2->class_id = INPUT_TREE;
  I2->w = 1;
  I2->input_index = 4;

  SK2->subtree[0] = I2;

  tree::ptr C1(new tree);
  C1->w = 1;
  C1->class_id = CONSTANT_TREE;
  C1->const_value = 0.2;

  K->subtree[0] = W;
  K->subtree[1] = C1;

  tree::ptr S(new tree);
  S->w = 1;
  S->class_id = UNARY_TREE;
  S->fname = "sigmoid";
  S->subtree.resize(1);

  tree::ptr W2(new tree);
  W2->w = 0.02;
  W2->class_id = WEIGHT_TREE;
  W2->subtree.resize(2);

  tree::ptr I3(new tree);
  I3->w = 1;
  I3->class_id = INPUT_TREE;
  I3->input_index = 1;

  tree::ptr C2(new tree);
  C2->w = -1;
  C2->class_id = CONSTANT_TREE;
  C2->const_value = 50;

  W2->subtree[0] = I3;
  W2->subtree[1] = C2;

  S->subtree[0] = W2;

  root->subtree[0] = K;
  root->subtree[1] = S;

  cout << "tree_evaluator::example_setup: complete with size " << root->count_trees() << endl;
}

void tree_evaluator::initialize(input_sampler sampler, int cdim) {
  stable = true;
  dim = cdim;
  learning_rate = fabs(rnorm(0, 0.1));
  weight_limit = u01(100, 10000);

  root = tree::ptr(new tree);
  root->initialize(dim, depth);
}

string tree_evaluator::status_report() const {
  stringstream ss;
  string sep = ",";
  ss << root->count_trees() << sep << learning_rate;
  return ss.str();
}
