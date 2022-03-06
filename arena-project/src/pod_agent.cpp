#include "pod_agent.hpp"

#include <sstream>

#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;

pod_agent::pod_agent() : agent() {
  class_id = POD_AGENT;
}

agent_ptr pod_agent::clone() const {
  shared_ptr<pod_agent> a(new pod_agent(*this));
  a->eval = eval->clone();
  a->csel = choice_selector_ptr(new choice_selector(*csel));
  a->parent_buf.clear();
  a->tstats = training_stats();
  a->optim_stats = optim_result<dvalue>();

  return a;
}
