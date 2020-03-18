#include <cassert>
#include <cmath>
#include <sstream>

#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

// tree evaluator

tree_evaluator::tree_evaluator(int d) : depth(d), evaluator() {}

hm<string, t_unary> tree_evaluator::unary_ops() {
  static hm<string, t_unary> unary_op;

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
    unary_op["signum"].f = [](double x) { return (x > 0) - (x < 0); };
    unary_op["signum"].fprime = [](double x) { return 0; };
  }

  return unary_op;
}

hm<string, t_binary> tree_evaluator::binary_ops() {
  static hm<string, t_binary> binary_op;

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

    binary_op["plus"].f = [](double a, double b) { return a + b; };
    binary_op["plus"].dfdx1 = [](double a, double b) { return 1; };
    binary_op["plus"].dfdx2 = [](double a, double b) { return 1; };

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

  return binary_op;
}

double tree_evaluator::complexity_penalty() const {
  int n = root->count_trees();
  return kernel(n, 1000);
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
  ss >> w >> cid;
  class_id = (tree_class)cid;

  if (class_id == CONSTANT_TREE) {
    ss >> const_value;
  } else if (class_id == INPUT_TREE) {
    ss >> input_index;
  } else {
    ss >> fname;
  }

  // create subtrees
  if (class_id == UNARY_TREE) {
    subtree.resize(1);
  } else if (class_id == BINARY_TREE) {
    subtree.resize(2);
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
        const_value = resbuf + rnorm(0, 0.1);
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

tree_evaluator::tree::ptr tree_evaluator::tree::clone() {
  tree::ptr t(new tree(*this));
  for (auto &s : t->subtree) s = s->clone();
  return t;
}

double tree_evaluator::tree::get_val(vec x) {
  double val;

  if (class_id == CONSTANT_TREE) {
    val = const_value;
  } else if (class_id == INPUT_TREE) {
    val = x[input_index];
  } else if (class_id == UNARY_TREE) {
    val = unary_ops()[fname].f(subtree[0]->evaluate(x));
  } else if (class_id == BINARY_TREE) {
    val = binary_ops()[fname].f(subtree[0]->evaluate(x), subtree[1]->evaluate(x));
  } else {
    throw runtime_error("Invalid tree class id!");
  }

  return resbuf = val;
}

double tree_evaluator::tree::evaluate(vec x) {
  return w * get_val(x);
}

void tree_evaluator::tree::initialize(int dim, int depth) {
  double p_cut = 1 / (double)(depth + 1);
  subtree.clear();
  w = rnorm();

  if (depth > 0 && u01() < (1 - p_cut)) {
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

void tree_evaluator::tree::apply_dw(double rescale) {
  w += dwbuf * rescale;
  assert(isfinite(w));

  ssw = pow(w, 2);
  for (auto t : subtree) {
    t->apply_dw(rescale);
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

void tree_evaluator::tree::update(vec input, double delta, double alpha, bool &stable) {
  ssw = 0;
  ssdw = 0;
  assert(isfinite(delta));

  if (class_id == BINARY_TREE) {
    double y1 = subtree[0]->resbuf;
    double y2 = subtree[1]->resbuf;
    double left_deriv = binary_ops()[fname].dfdx1(y1, y2);
    double right_deriv = binary_ops()[fname].dfdx2(y1, y2);
    subtree[0]->update(input, delta, w * left_deriv * alpha, stable);
    subtree[1]->update(input, delta, w * right_deriv * alpha, stable);
  } else if (class_id == UNARY_TREE) {
    double y = subtree[0]->resbuf;
    double deriv = unary_ops()[fname].fprime(y);
    subtree[0]->update(input, delta, w * deriv * alpha, stable);
  }

  double dydw = alpha * resbuf;
  if (!isfinite(dydw)) stable = false;

  for (auto t : subtree) {
    ssw += t->ssw;
    ssdw += t->ssdw;
  }
  ssw += pow(w, 2);

  if (fabs(dydw) > 1e-6 && isfinite(dydw)) {
    ssdw += pow(dwbuf = delta / dydw, 2);
  } else {
    dwbuf = 0;
  }
}

evaluator_ptr tree_evaluator::clone() const {
  shared_ptr<tree_evaluator> e(new tree_evaluator(*this));
  e->root = root->clone();
  return e;
}

double tree_evaluator::evaluate(vec x) {
  return root->evaluate(x);
}

void tree_evaluator::update(vec input, double output, int age) {
  double current = evaluate(input);

  // compute weight increments
  root->update(input, output - current, 1, stable);

  double wlen = sqrt(root->ssw);
  double dwlen = sqrt(root->ssdw);
  double time_scale = 1 / (log(age + 1) + 1);
  double rate = learning_rate * time_scale;
  double limit = 0.01 * time_scale;

  if (rate * dwlen > limit * wlen) {
    rate = limit * wlen / dwlen;
  }

  root->apply_dw(rate);

  // scale down weights if too large
  wlen = sqrt(root->ssw);
  if (wlen > weight_limit) {
    root->scale_weights(weight_limit / wlen);
    assert(sqrt(root->ssw) <= 1.1 * weight_limit);
  }
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

void tree_evaluator::initialize(input_sampler sampler, int cdim) {
  stable = true;
  vec input = sampler();
  dim = input.size();
  learning_rate = fabs(rnorm(0, 0.2));
  weight_limit = u01(1, 10000);

  root = tree::ptr(new tree);
  root->initialize(dim, depth);
}

string tree_evaluator::status_report() const {
  stringstream ss;
  string sep = ",";
  ss << root->count_trees() << sep << learning_rate;
  return ss.str();
}
