#include "simple_pod_evaluator.hpp"
#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;
using namespace pod_game_parameters;

// return basic on-track feature
double simple_pod_evaluator::evaluate(vec x) {
  double thrust = x[1];
  double a_ncp = x[4];
  double c_angle = x[0];
  bool same_sign = signum(a_ncp) == signum(c_angle);
  double res = thrust > 90 && same_sign;
}

bool simple_pod_evaluator::update(vec input, double output, int age) { return true; }

evaluator_ptr simple_pod_evaluator::mate(evaluator_ptr partner) const { return evaluator_ptr(new simple_pod_evaluator); }
evaluator_ptr simple_pod_evaluator::mutate() const { return evaluator_ptr(new simple_pod_evaluator); }
evaluator_ptr simple_pod_evaluator::clone() const { return evaluator_ptr(new simple_pod_evaluator); }
std::string simple_pod_evaluator::serialize() const { return "simple_pod_evaluator"; }
void simple_pod_evaluator::deserialize(std::stringstream &data) {}
void simple_pod_evaluator::initialize(input_sampler sampler, int cdim) {}
std::string simple_pod_evaluator::status_report() const { return "dummy status"; }
double simple_pod_evaluator::complexity_penalty() const { return 0; }
double simple_pod_evaluator::complexity() const { return 0; }
