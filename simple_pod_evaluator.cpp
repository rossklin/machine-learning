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
  double dist = x[9];
  bool boost = x[2] > 0;

  double target_angle = signum(a_ncp) * fmin(abs(a_ncp), angular_speed);
  double angle_match = kernel(angle_difference(c_angle, target_angle), angular_speed / 3);

  double target_thrust = 100 * angle_match;
  double thrust_match = kernel(thrust - target_thrust, 20);

  bool want_boost = dist > 4000 && angle_match > 0.95;
  double boost_match = boost == want_boost;

  return angle_match * thrust_match + boost_match;
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

set<int> simple_pod_evaluator::list_inputs() const {
  return {0, 1, 2, 4, 9};
}