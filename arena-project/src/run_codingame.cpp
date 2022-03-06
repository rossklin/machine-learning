#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pod_agent.hpp"
#include "pod_game.hpp"
#include "pod_game_generator.hpp"
#include "utility.hpp"

using namespace std;

void run_codingame(string brain) {
  // create agent
  stringstream ss(brain);
  agent_ptr a = deserialize_agent(ss);
  a->csel->set_exploration_rate(0.1);

  // create game object
  game_generator_ptr ggen(new pod_game_generator(2, 2, [a]() -> agent_ptr { return a->clone(); }));
  game_ptr g_base = ggen->generate_starting_state(ggen->make_teams({a, a}));
  shared_ptr<pod_game> g = static_pointer_cast<pod_game>(g_base);

  // read checkpoints from cin
  g->setup_from_input(cin);
  int ncp = g->checkpoint.size();

  while (true) {
    for (auto x : g->typed_agents) {
      pod_agent::ptr a = x.second;
      pod_data &pod = a->data;
      int next_cp;
      int degrees;
      cin >> pod.x.x >> pod.x.y >> pod.v.x >> pod.v.y >> degrees >> next_cp;
      pod.a = (2 * M_PI * degrees) / 360;
      pod.passed_checkpoint = modulo(next_cp - 1, ncp);
      pod.lap += pod.passed_checkpoint != pod.previous_checkpoint && pod.passed_checkpoint == 0;
      pod.previous_checkpoint = pod.passed_checkpoint;
    }

    for (auto x : g->typed_agents) {
      pod_agent::ptr a = x.second;
      if (a->team != 0) continue;

      shared_ptr<pod_choice> c = static_pointer_cast<pod_choice>(a->select_choice(g));
      point target = a->data.x + 100 * normv(c->angle + a->data.a);
      stringstream ss;

      ss << (int)target.x << sep << (int)target.y << sep;

      if (c->boost) {
        ss << "BOOST";
        a->data.boost_count = 0;
      } else if (c->shield) {
        ss << "SHIELD";
        a->data.shield_active = 3;
      } else {
        ss << (int)c->thrust;
      }

      if (a->data.shield_active > 0) a->data.shield_active--;

      cout << ss.str() << endl;
    }
  }
}

int main() {
  run_codingame(agent_str);
  return 0;
}
