#pragma once

#include <omp.h>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

const std::string sep = " ";

struct MutexType {
  MutexType();
  ~MutexType();
  void Lock();
  void Unlock();

  MutexType(const MutexType &);
  MutexType &operator=(const MutexType &);

 public:
  omp_lock_t lock;
};

template <typename T = double, typename V = double>
std::vector<V> map(std::function<V(T)> f, std::vector<T> x) {
  std::vector<V> res(x.size());
  for (int i = 0; i < x.size(); i++) res[i] = f(x[i]);
  return res;
}

template <typename K, typename V>
std::vector<K> hm_keys(hm<K, V> x) {
  std::vector<K> res;
  for (auto y : x) res.push_back(y.first);
  return res;
}

template <typename K, typename V>
std::vector<K> hm_values(hm<K, V> x) {
  std::vector<K> res;
  for (auto y : x) res.push_back(y.second);
  return res;
}

point operator+(const point &a, const point &b);
point operator-(const point &a, const point &b);
point operator*(const double &s, const point &a);

std::ostream &operator<<(std::ostream &os, const point &x);

double distance(point a, point b);

double point_angle(point p);

point truncate_point(point x);

double scalar_mult(point a, point b);

double sproject(point a, point r);

point normv(double a);
;

point normalize(point x);

template <typename T>
T modulo(T x, T p) {
  int num = floor(x / (double)p);
  return x - num * p;
}

double angle_difference(double a, double b);

double time_discount(double x, double t);

double u01(double a = 0, double b = 1);

double rnorm(double m = 0, double s = 1);

int rand_int(int a, int b);

double signum(double x);

double sigmoid(double x, double h);

// std::vector arithmetics
vec operator+(vec a, const vec &b);

vec operator-(vec a, const vec &b);

vec operator*(const double &s, vec a);

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &x) {
  os << x.size();
  if (x.empty()) return os;
  for (auto y : x) os << " " << y;
  return os;
};

std::istream &operator>>(std::istream &os, vec &x);

double l2norm(vec x);

double kernel(double x, double h);

template <typename T>
std::set<T> set_difference(std::set<T> a, std::set<T> b) {
  for (auto x : b) a.erase(x);
  return a;
}

template <typename T>
std::set<T> set_union(std::set<T> a, std::set<T> b) {
  a.insert(b.begin(), b.end());
  return a;
}

template <typename T>
std::set<T> set_intersect(std::set<T> a, std::set<T> b) {
  return set_difference(a, set_difference(a, b));
}

template <typename T>
std::set<T> set_symdiff(std::set<T> a, std::set<T> b) {
  return set_difference(set_union(a, b), set_intersect(a, b));
}

std::vector<int> seq(int a, int b);

double cat(std::function<double(double, double)> f, vec x);

double quantile(vec x, double r);

double fminx(double a, double b);
double fmaxx(double a, double b);
double min(vec x);
double max(vec x);
double sum(vec x);
double mean(vec x);
bool has_nan(vec x);
int max_idx(vec x);

template <typename K, typename V>
K best_key(std::vector<K> keys, std::function<double(K)> eval) {
  double best = -INFINITY;
  auto res = keys.end();
  for (auto k = keys.begin(); k != keys.end(); k++) {
    double y = eval(*k);
    if (y > best) {
      best = y;
      res = k;
    }
  }

  assert(res != keys.end());
  return *res;
}

double stdev(vec x);

template <typename F, typename T>
std::vector<T> type_shift(std::vector<F> x) {
  std::vector<T> res(x.size());
  for (int i = 0; i < x.size(); i++) res[i] = x[i];
  return res;
}

template <typename T>
T hash_sample(hm<int, T> x) {
  T def = 0;
  for (auto y = x.begin(); y != x.end(); y++) {
    if (u01() <= 1 / (double)x.size()) {
      return y->second;
    } else {
      x.erase(y--);
    }
  }
  // todo: log arithmetics warning
  return def;
}

template <typename T>
std::vector<T> vector_sample(std::vector<T> data, int n) {
  std::vector<T> res;
  while (n > 0) {
    int idx = rand_int(0, data.size() - 1);
    res.push_back(data[idx]);
    data.erase(data.begin() + idx);
    n--;
  }
  return res;
}

template <typename T>
T sample_one(std::vector<T> data) {
  return data[rand_int(0, data.size() - 1)];
}

template <typename T>
T ranked_sample(std::vector<T> x, double q) {
  assert(!x.empty());
  while (true)
    for (auto y : x)
      if (u01() < q) return y;
}
