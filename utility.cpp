#include <cassert>
#include <cfloat>
#include <cmath>
#include <random>
#include <sstream>

#include "types.hpp"
#include "utility.hpp"

using namespace std;

point operator+(const point &a, const point &b) { return {a.x + b.x, a.y + b.y}; };
point operator-(const point &a, const point &b) { return {a.x - b.x, a.y - b.y}; };
point operator*(const double &s, const point &a) { return {s * a.x, s * a.y}; };

ostream &operator<<(ostream &os, const point &x) { return os << x.x << " " << x.y; };

double distance(point a, point b) {
  double d1 = a.x - b.x;
  double d2 = a.y - b.y;
  return sqrt(d1 * d1 + d2 * d2);
}

double point_angle(point p) {
  if (p.x > 0) {
    return atan(p.y / p.x);
  } else if (p.x < 0) {
    return M_PI + atan(p.y / p.x);
  } else if (p.y > 0) {
    return M_PI / 2;
  } else {
    return -M_PI / 2;
  }
}

point truncate_point(point x) {
  return {floor(x.x), floor(x.y)};
}

double scalar_mult(point a, point b) {
  return a.x * b.x + a.y * b.y;
}

double sproject(point a, point r) {
  return scalar_mult(a, r) / scalar_mult(r, r);
}

point normv(double a) {
  return {cos(a), sin(a)};
}

point normalize(point x) {
  return 1 / distance({0, 0}, x) * x;
}

double angle_difference(double a, double b) {
  return modulo(a - b + M_PI, 2 * M_PI) - M_PI;
}

double time_discount(double x, double t) {
  return x * exp(-t / 4);
}

default_random_engine get_random_engine() {
  static default_random_engine random_generator(time(NULL));
  return random_generator;
}

double u01(double a, double b) {
  uniform_real_distribution<double> distribution(a, b);
  auto generator = bind(distribution, get_random_engine());
  return generator();
}

double rnorm(double m, double s) {
  normal_distribution<double> distribution(m, s);
  auto generator = bind(distribution, get_random_engine());
  return generator();
}

int rand_int(int a, int b) {
  assert(b >= a);
  return u01(a, b + 1 - DBL_EPSILON);
}

double signum(double x) {
  return (x > 0) - (x < 0);
}

double sigmoid(double x, double h) {
  return 2 / (1 + exp(-x / h)) - 1;
}

// vector arithmetics
vec operator+(vec a, const vec &b) {
  for (int i = 0; i < a.size(); i++) a[i] += b[i];
  return a;
};

vec operator-(vec a, const vec &b) {
  for (int i = 0; i < a.size(); i++) a[i] -= b[i];
  return a;
};

vec operator*(const double &s, vec a) {
  for (int i = 0; i < a.size(); i++) a[i] *= s;
  return a;
};

istream &operator>>(istream &os, vec &x) {
  int n;
  os >> n;
  x.resize(n);
  for (auto &y : x) os >> y;
  return os;
};

double l2norm(vec x) {
  double s = 0;
  for (auto y : x) s += y * y;
  return sqrt(s);
};

double kernel(double x, double h) {
  return exp(-pow(x / h, 2));
}

vector<int> seq(int a, int b) {
  int n = b - a + 1;
  assert(n > 0);

  vector<int> idx(n);
  iota(idx.begin(), idx.end(), a);
  return idx;
}

double cat(function<double(double, double)> f, vec x) {
  assert(x.size() > 0);
  double y = x[0];
  for (int i = 1; i < x.size(); i++) y = f(y, x[i]);
  return y;
}

double quantile(vec x, double r) {
  sort(x.begin(), x.end());
  double n = x.size();
  return x[(int)(r * n)];
}

double fminx(double a, double b) { return fmin(a, b); }
double fmaxx(double a, double b) { return fmax(a, b); }
double min(vec x) { return cat(fminx, x); }
double max(vec x) { return cat(fmaxx, x); }
double sum(vec x) {
  return cat([](double a, double b) { return a + b; }, x);
}
double mean(vec x) { return sum(x) / x.size(); }
bool has_nan(vec x) { return isnan(sum(x)); }
int max_idx(vec x) {
  int idx = -1;
  double best = -INFINITY;
  for (int i = 0; i < x.size(); i++) {
    if (x[i] > best) {
      best = x[i];
      idx = i;
    }
  }

  return idx;
}

double stdev(vec x) {
  double m = mean(x);
  function<double(double)> f = [m](double x) -> double { return pow(x - m, 2); };
  return sqrt(sum(map(f, x)));
}
