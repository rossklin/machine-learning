#include "pod_agent.hpp"
#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;

agent_ptr pod_agent::clone() const {
  shared_ptr<pod_agent> a(new pod_agent);
  a->eval = eval->clone();
  a->csel = csel;
  a->data = data;
  return a;
}
