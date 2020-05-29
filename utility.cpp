#include "utility.hpp"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <iterator>
#include <random>
#include <sstream>

#include "types.hpp"

using namespace std;

MutexType::MutexType() { omp_init_lock(&lock); }
MutexType::~MutexType() { omp_destroy_lock(&lock); }
void MutexType::Lock() { omp_set_lock(&lock); }
void MutexType::Unlock() { omp_unset_lock(&lock); }

MutexType::MutexType(const MutexType &) { omp_init_lock(&lock); }
MutexType &MutexType::operator=(const MutexType &) { return *this; }

double numgrad(double x, double h, function<double(double)> f) {
  double a = f(x - h / 2);
  double b = f(x + h / 2);
  return (b - a) / h;
}

string join_string(const vector<string> vec, string delim) {
  if (vec.empty()) return "";

  stringstream ss;
  for (int i = 0; i < vec.size() - 1; i++) ss << vec[i] << delim;
  ss << vec.back();

  return ss.str();
}

vec voptim(vec x, function<double(vec)> fopt, function<vec(vec)> fgrad) {
  double xlim = 2e-2 * l2norm(x);
  double border = 1e3 * l2norm(x);
  vec d = fgrad(x);
  int max_its = 40;
  double y = fopt(x);
  double y_last = 2 * y;
  double rel_lim = 1e-2;
  int i;
  double step_lim = 0.1;
  int n = x.size();
  // double h = 1e-6 * l2norm(x);
  vec x_last = x;

  vec ys = {y};
  for (i = 0; i < max_its && l2norm(d) > xlim && l2norm(x) < border && (y_last - y) / y > rel_lim; i++) {
    d = fgrad(x);

    if (l2norm(d) == 0 || y == 0) break;

    // for (int j = 0; j < n; j++) {
    //   vec x2 = x, x1 = x;
    //   x2[j] += h / 2;
    //   x1[j] -= h / 2;
    //   double d2j = (fgrad(x2)[j] - fgrad(x1)[j]) / h;
    // }

    // break at zero criteria (lesser of gradient descent and newton's root solver)
    auto f0c = [y](double d) -> double {
      if (d > 0) {
        d = min(d, y / d);
      } else {
        d = max(d, y / d);
      }
      return d;
    };

    d = map<double, double>(f0c, d);

    // ds.push_back(d);

    // do not allow stepping more than X part of state
    if (l2norm(d) > step_lim * l2norm(x)) d = step_lim * l2norm(x) / l2norm(d) * d;

    // // do not allow stepping so that gradient will change by more than X part
    // if (l2norm(d2) > 0 && l2norm(d) > step_lim * l2norm(d2)) d = step_lim * l2norm(x) / l2norm(d2) * d;

    x_last = x;
    x = x - d;

    y_last = y;
    y = fopt(x);
    ys.push_back(y);
  }

  // cout << ys << endl;

  return y < y_last ? x : x_last;
}

// minimize f > 0
double foptim(double x, function<double(double)> f) {
  double xlim = 2e-2 * fabs(x);
  double h = 1e-2 * xlim;
  double edge = 1e3 * fabs(x);
  double d = 2 * xlim;
  int max_its = 40;
  double y_last = 0;
  double y = f(x);
  double rel_lim = 1e-2;
  int i;

  // vector<double> xs, ys, ds;
  for (i = 0; i < max_its && fabs(d) > xlim && x > 0 && x < edge && fabs(y - y_last) / y > rel_lim; i++) {
    d = numgrad(x, h, f);
    // xs.push_back(x);
    // ys.push_back(y);

    if (d == 0 || y == 0) break;

    if (d > 0) {
      d = min(d, y / d);
    } else {
      d = max(d, y / d);
    }

    // ds.push_back(d);

    x -= d;

    // don't step further than to zero - we have extreme gradients but assume f > 0
    y_last = y;
    y = f(x);
  }

  // string sxs = join_string(map<double, string>([](double x) -> string { return to_string(x); }, xs), ",");
  // string sys = join_string(map<double, string>([](double x) -> string { return to_string(x); }, ys), ",");
  // string sds = join_string(map<double, string>([](double x) -> string { return to_string(x); }, ds), ",");

  // cout << "optpath <- list( xs = c(" << sxs << "), ys = c(" << sys << "), ds = c(" << sds << "))" << endl;

  return x;
}

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

mt19937 &get_random_engine() {
  // static default_random_engine random_generator(time(NULL));
  static random_device rd;   //Will be used to obtain a seed for the random number engine
  static mt19937 gen(rd());  //Standard mersenne_twister_engine seeded with rd()
  return gen;
}

double u01(double a, double b) {
  uniform_real_distribution<double> distribution(a, b);
  mt19937 &gen = get_random_engine();
  MutexType m;
  m.Lock();
  double test = distribution(gen);
  m.Unlock();
  return test;
}

double rnorm(double m, double s) {
  normal_distribution<double> distribution(m, s);
  mt19937 &gen = get_random_engine();
  MutexType mut;
  mut.Lock();
  double test = distribution(gen);
  mut.Unlock();
  return test;
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

ostream &operator<<(ostream &os, const dvalue &x) {
  return os << x.current << sep << x.last;
}

istream &operator>>(istream &is, dvalue &x) {
  return is >> x.current >> x.last;
}

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
