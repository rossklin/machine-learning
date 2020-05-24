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

enum unary_ops {
  UNARY_SIN,
  UNARY_COS,
  UNARY_ATAN,
  UNARY_SIGMOID,
  UNARY_ABS,
  UNARY_NUM
};

enum binary_ops {
  BINARY_KERNEL,
  BINARY_PRODUCT,
  BINARY_NUM
};

vector<t_unary> unary_op;
vector<t_binary> binary_op;

void init_ops() {
  static bool init = false;
  if (init) return;

  MutexType m;
  m.Lock();

  cout << "Running init ops" << endl;

  unary_op.resize(UNARY_NUM);

  unary_op[UNARY_SIN].f = [](double x) { return sin(x); };
  unary_op[UNARY_SIN].fprime = [](double x) { return cos(x); };
  unary_op[UNARY_COS].f = [](double x) { return cos(x); };
  unary_op[UNARY_COS].fprime = [](double x) { return -sin(x); };
  unary_op[UNARY_ATAN].f = [](double x) { return atan(x); };
  unary_op[UNARY_ATAN].fprime = [](double x) { return 1 / (1 + pow(x, 2)); };
  unary_op[UNARY_SIGMOID].f = [](double x) { return 1 / (1 + exp(-x)); };
  unary_op[UNARY_SIGMOID].fprime = [](double x) -> double {
    if (fabs(x) > 20) {
      return 0;
    } else {
      return exp(-x) / pow(1 + exp(-x), 2);
    }
  };
  unary_op[UNARY_ABS].f = [](double x) { return fabs(x); };
  unary_op[UNARY_ABS].fprime = [](double x) { return (x > 0) - (x < 0); };

  binary_op.resize(BINARY_NUM);

  binary_op[BINARY_KERNEL].f = [](double x, double h) -> double {
    if (h > 0 && pow(x / h, 2) < 40) {
      return exp(-pow(x / h, 2));
    } else {
      return 0;
    }
  };

  binary_op[BINARY_KERNEL].dfdx1 = [](double x, double h) -> double {
    if (h > 0 && pow(x / h, 2) < 40) {
      return -2 * x / pow(h, 2) * exp(-pow(x / h, 2));
    } else {
      return 0;
    }
  };

  binary_op[BINARY_KERNEL].dfdx2 = [](double x, double h) -> double {
    if (h > 0 && pow(x / h, 2) < 40) {
      return 2 * pow(x, 2) / pow(h, 3) * exp(-pow(x / h, 2));
    } else {
      return 0;
    }
  };

  binary_op[BINARY_PRODUCT].f = [](double a, double b) { return a * b; };
  binary_op[BINARY_PRODUCT].dfdx1 = [](double a, double b) { return b; };
  binary_op[BINARY_PRODUCT].dfdx2 = [](double a, double b) { return a; };

  m.Unlock();

  init = true;
}

tree_evaluator::tree_evaluator() : evaluator() {
  init_ops();
  gamma = fabs(rnorm(0.01, 0.005));
  tag = "tree";
}

double tree_evaluator::complexity() const { return root->count_trees(); }

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
void tree_evaluator::tree::mutate(int dim, evaluator::dist_category dc) {
  const vector<double> change = {2e-3, 1e-2, 5e-2};
  w += rnorm(0, change[dc]);

  double p_grow = change[dc];
  double p_reduce = change[dc];

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
      for (auto t : subtree) t->mutate(dim, dc);
    }
  } else {
    if (u01() < p_grow) {
      // extend tree
      if (u01() < 0.2) {
        // unary
        class_id = WEIGHT_TREE;
        subtree.resize(rand_int(2, max_weighted_subtrees));
      } else if (u01() < 0.7) {
        // unary
        class_id = UNARY_TREE;
        fname = rand_int(0, UNARY_NUM - 1);
        subtree.resize(1);
      } else {
        // binary
        class_id = BINARY_TREE;
        fname = rand_int(0, BINARY_NUM - 1);
        subtree.resize(2);
      }

      for (auto &t : subtree) {
        int n = min(dim, ranked_sample(seq(2, 5), 0.8));
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
    val = unary_op[fname].f(subtree[0]->evaluate(x));
  } else if (class_id == BINARY_TREE) {
    val = binary_op[fname].f(subtree[0]->evaluate(x), subtree[1]->evaluate(x));
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
    if (u01() < 0.4) {
      // weight
      class_id = WEIGHT_TREE;
      int size = ranked_sample(seq(2, max_weighted_subtrees), 0.5);
      subtree.resize(size);
    } else if (u01() < 0.6) {
      // unary
      class_id = UNARY_TREE;
      fname = rand_int(0, UNARY_NUM - 1);
      subtree.resize(1);
    } else {
      // binary
      class_id = BINARY_TREE;
      fname = rand_int(0, BINARY_NUM - 1);
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

  if (subtree.empty()) return;

  vector<vector<int>> parts = random_partition<int>(inputs, subtree.size());

  for (int i = 0; i < subtree.size(); i++) {
    subtree[i] = tree::ptr(new tree);
    subtree[i]->initialize(parts[i]);
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
    double left_deriv = binary_op[fname].dfdx1(y1, y2);
    double right_deriv = binary_op[fname].dfdx2(y1, y2);
    offset = subtree[0]->calculate_dw(dgdw, offset, delta, w * left_deriv * alpha, gamma);
    offset = subtree[1]->calculate_dw(dgdw, offset, delta, w * left_deriv * alpha, gamma);
  } else if (class_id == UNARY_TREE) {
    double y = subtree[0]->resbuf;
    double deriv = unary_op[fname].fprime(y);
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

vec tree_evaluator::gradient(vec input, double target, double w_reg) const {
  vec x = root->get_weights();
  double output = root->evaluate(input);
  double delta = target - output;
  root->calculate_dw(x, 0, delta, 1, w_reg);
  return x;
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

evaluator_ptr tree_evaluator::mate(evaluator_ptr partner_buf) const {
  shared_ptr<tree_evaluator> partner = static_pointer_cast<tree_evaluator>(partner_buf);
  shared_ptr<tree_evaluator> child = static_pointer_cast<tree_evaluator>(clone());
  child->learning_rate = fmax(rnorm(0.5, 0.1) * (learning_rate + partner->learning_rate), 1e-5);
  child->weight_limit = fmax(rnorm(0.5, 0.1) * (weight_limit + partner->weight_limit), 1);

  tree::ptr sub = partner->root->get_subtree(0.3);
  child->root->emplace_subtree(sub, 0.3);

  return child;
}

evaluator_ptr tree_evaluator::mutate(evaluator::dist_category dc) const {
  if (dc == MUT_RANDOM) dc = sample_one<dist_category>({MUT_SMALL, MUT_MEDIUM, MUT_LARGE});
  shared_ptr<tree_evaluator> child = static_pointer_cast<tree_evaluator>(clone());
  child->root->mutate(dim, dc);

  vector<double> spread = {1e-3, 1e-2, 1e-1};
  child->learning_rate = fmax(child->learning_rate + rnorm(0, spread[dc]), 1e-5);
  child->weight_limit = fmax(child->weight_limit * rnorm(1, spread[dc]), 1);
  child->gamma = fmax(child->gamma + rnorm(0, 1e-1 * spread[dc]), 0);
  child->mut_tag = dc;
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

void tree_evaluator::set_weights(const vec &w) {
  root->set_weights(w);
}

vec tree_evaluator::get_weights() const {
  return root->get_weights();
}

void tree_evaluator::initialize(input_sampler sampler, int cdim, set<int> ireq) {
  stable = true;
  dim = cdim;
  learning_rate = fabs(rnorm(0, 0.5));
  weight_limit = u01(100, 10000);

  // prepare input indices for the tree to use
  vector<int> ibuf;
  for (int i = 0; i < cdim; i++) {
    int n = ranked_sample(seq(0, 3), 0.75);
    if (ireq.count(i)) n = ranked_sample(seq(1, 3), 0.9);
    ibuf.insert(ibuf.end(), n, i);
  }
  random_shuffle(ibuf.begin(), ibuf.end());

  root = tree::ptr(new tree);
  root->initialize(ibuf);
}

string tree_evaluator::status_report() const {
  stringstream ss;
  ss << root->count_trees() << comma << learning_rate;
  return ss.str();
}

set<int> tree_evaluator::list_inputs() const {
  return root->list_inputs();
}

void tree_evaluator::add_inputs(set<int> inputs) {
  root->add_inputs({inputs.begin(), inputs.end()});
}