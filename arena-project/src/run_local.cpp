#include "ml_pod.cpp"

void run_local(vector<string> brain) {
  P(pod_game)
  g(new pod_game);

  // initialize players
  int n = brain.size();
  vector<P(agent)> player(2 * n);
  for (int i = 0; i < n; i++) {
    player[i] = pod_game::tree_pod_agent();
    stringstream ss(brain[i]);
    player[i]->cval->deserialize(ss);
    player[i]->team = 1;
    player[i]->team_index = i;
  }
  for (int i = n; i < 2 * n; i++) {
    player[i] = pod_game::simple_pod_agent();
    player[i]->team = 2;
  }

  g = static_pointer_cast<pod_game>(pod_game::generate_starting_state(player));
  g->enable_output = true;
  g->play();
}

string p0 = "";

string p1 = "";

int main() {
  choice::tree_evaluator::setup();
  run_local({p0, p1});
  return 0;
}
