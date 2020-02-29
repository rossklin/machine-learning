#include "pod_agent.hpp"
#include "pod_game.hpp"
#include "utility.hpp"

using namespace std;

template <typename E>
pod_agent<E>::pod_agent() : agent<E>() {
  label = "tree-pod-agent";
}

template <typename E>
typename agent<E>::ptr pod_agent<E>::clone() {
  pod_agent<E>::ptr a(new pod_agent<E>);
  a->eval = eval->clone();
  a->csel = csel;
  a->data = data;
  return a;
}
