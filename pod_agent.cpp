#include "pod_agent.hpp"
#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;

template <typename E>
pod_agent<E>::pod_agent() : agent<E>() {
  label = "tree-pod-agent";
}

agent_ptr pod_agent::clone() {
  pod_agent_ptr a = pod_agent_ptr(new pod_agent(eval, csel));
  a->data = data;
  return a;
}

// return basic on-track feature
double simple_pod_evaluator::evaluate(std::vector<double> x) {
  double a_ncp = x[4];
  double c_angle = x[0];
  double da = a_ncp - c_angle;
  bool same_sign = signum(a_ncp) == signum(c_angle);
  bool matches_aspeed = fabs(a_ncp) > pod_game::angular_speed && fabs(fabs(c_angle) - pod_game::angular_speed) < 0.01;
  bool on_target = fabs(a_ncp) <= pod_game::angular_speed && fabs(a_ncp - c_angle) < pod_game::angular_speed / 3;
  return same_sign && (matches_aspeed || on_target);
}

void simple_pod_evaluator::update(std::vector<double> input, double output, int age) {}

evaluator_ptr simple_pod_evaluator::mate(evaluator_ptr partner) { return evaluator_ptr(new simple_pod_evaluator); }
evaluator_ptr simple_pod_evaluator::mutate() { return evaluator_ptr(new simple_pod_evaluator); }
evaluator_ptr simple_pod_evaluator::clone() { return evaluator_ptr(new simple_pod_evaluator); }
std::string simple_pod_evaluator::serialize() { return "simple_pod_evaluator"; }
void simple_pod_evaluator::deserialize(std::string data) {}
void simple_pod_evaluator::initialize(input_sampler sampler, int cdim) {}
std::string simple_pod_evaluator::status_report() { return "dummy status"; }
