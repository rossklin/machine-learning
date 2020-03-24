#include <sstream>

#include "pod_agent.hpp"
#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;

pod_agent::pod_agent() : agent() {
  class_id = POD_AGENT;
}

agent_ptr pod_agent::clone() const {
  shared_ptr<pod_agent> a(new pod_agent);
  a->eval = eval->clone();
  a->csel = csel;
  a->data = data;
  a->label = label;
  a->class_id = class_id;

  return a;
}
