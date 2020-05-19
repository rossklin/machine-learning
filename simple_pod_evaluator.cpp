#include "simple_pod_evaluator.hpp"

#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;
using namespace pod_game_parameters;

simple_pod_evaluator::simple_pod_evaluator() : evaluator() {}

void simple_pod_evaluator::set_learning_rate(double r) {}

// return basic on-track feature
double simple_pod_evaluator::evaluate(vec x) {
  double thrust = x[1];
  double a_ncp = x[9];
  double c_angle = x[0];
  double dist = x[10];
  bool boost = x[2] > 0;

  double target_angle = signum(a_ncp) * fmin(fabs(a_ncp), angular_speed);
  double angle_match = kernel(angle_difference(c_angle, target_angle), angular_speed / 3);

  double target_thrust = 100 * angle_match;
  double thrust_match = kernel(thrust - target_thrust, 20);

  bool want_boost = dist > 4000 && angle_match > 0.95;
  double boost_match = boost == want_boost;

  return angle_match * thrust_match + boost_match;
}

bool simple_pod_evaluator::update(vector<record> results, int age, int mut_age, double &rel_change) {
  rel_change = 0;
  return true;
}

void simple_pod_evaluator::prune(double l) {}

evaluator_ptr simple_pod_evaluator::mate(evaluator_ptr partner) const { return evaluator_ptr(new simple_pod_evaluator); }
evaluator_ptr simple_pod_evaluator::mutate(evaluator::dist_category dc) const { return evaluator_ptr(new simple_pod_evaluator); }
evaluator_ptr simple_pod_evaluator::clone() const { return evaluator_ptr(new simple_pod_evaluator); }
std::string simple_pod_evaluator::serialize() const { return "simple_pod_evaluator"; }
void simple_pod_evaluator::deserialize(std::stringstream &data) {}
void simple_pod_evaluator::initialize(input_sampler sampler, int cdim, set<int> ireq) {}
std::string simple_pod_evaluator::status_report() const { return "dummy status"; }
double simple_pod_evaluator::complexity() const { return 0; }

set<int> simple_pod_evaluator::list_inputs() const {
  vector<int> buf = seq(0, 100);
  return set<int>(buf.begin(), buf.end());
}

void simple_pod_evaluator::add_inputs(set<int> inputs) {}