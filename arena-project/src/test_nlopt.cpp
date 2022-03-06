#include <iostream>
#include <nlopt.hpp>

#include "utility.hpp"

using namespace std;
double nlopt_f(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  double inner = x[0] * (x[1] - 1) + x[1];
  double f = pow(inner, 2);
  grad[0] = 2 * (x[1] - 1) * inner;
  grad[1] = 2 * (x[0] + 1) * inner;
  cout << "Objective: " << f << endl;
  return f;
}

int main() {
  nlopt::opt opt(nlopt::LD_LBFGS, 2);
  opt.set_min_objective(nlopt_f, NULL);
  opt.set_xtol_rel(1e-2);
  double minf;
  vector<double> x0 = {3, 3};
  vector<double> x = x0;
  double y = 3 * (3 - 1) + 3;

  try {
    nlopt::result result = opt.optimize(x, minf);
    cout << "found minimum " << minf << endl;
    cout << "new x: " << x << endl;
    cout << "change y: from " << y << " to " << nlopt_f(x, x0, NULL) << endl;
  } catch (std::exception &e) {
    cout << "nlopt failed: " << e.what() << endl;
  }

  return 0;
}