#include "game_generator.hpp"
#include "agent.hpp"
#include "game.hpp"
#include "utility.hpp"

using namespace std;

template <typename GAME_CLASS, typename REFBOT_CLASS>
game_generator<GAME_CLASS, REFBOT_CLASS>::game_ptr game_generator<GAME_CLASS, REFBOT_CLASS>::generate_starting_state(std::vector<agent_ptr> p) {
  player_table pl;
  for (auto a : p) pl[a->id] = a;
  shared_ptr<GAME_CLASS> g(new pod_game(pl));
  g->initialize();
  return g;
}

template <typename GAME_CLASS, typename REFBOT_CLASS>
vector<typename game_generator<GAME_CLASS, REFBOT_CLASS>::agent_ptr> game_generator<GAME_CLASS, REFBOT_CLASS>::make_teams(vector<agent_ptr> ps) {
  vector<agent_ptr> buf(nr_of_teams * ppt);

  for (int filter_tid = 0; filter_tid < ps.size(); filter_tid++) {
    agent_ptr b = ps[filter_tid];
    b->team = filter_tid;
    for (int k = 0; k < ppt; k++) {
      agent_ptr a = b->clone();
      a->team = filter_tid;
      a->team_index = k;
      buf[filter_tid * ppt + k] = a;
    }
  }

  return buf;
}

template <typename GAME_CLASS, typename REFBOT_CLASS>
game_generator<GAME_CLASS, REFBOT_CLASS>::game_ptr game_generator<GAME_CLASS, REFBOT_CLASS>::team_bots_vs(agent_ptr a) {
  vector<agent_ptr> ps(nr_of_teams);
  ps[0] = a;
  for (int i = 1; i < nr_of_teams; i++) ps[i] = new REFBOT_CLASS;
  ps = make_teams(ps);
  return generate_starting_state(ps);
}

template <typename GAME_CLASS, typename REFBOT_CLASS>
int game_generator<GAME_CLASS, REFBOT_CLASS>::choice_dim() {
  game_ptr g = team_bots_vs(new REFBOT_CLASS);
  agent_ptr p = g->players.begin()->second;
  choice_ptr c = p->select_choice(g);
  vec input = g->vectorize_choice(c, p->id);
  return input.size();
}

template <typename GAME_CLASS, typename REFBOT_CLASS>
function<vec()> game_generator<GAME_CLASS, REFBOT_CLASS>::generate_input_sampler() {
  game_ptr g = team_bots_vs(new REFBOT_CLASS);
  auto buf = g->play(0);

  return [buf]() -> vec {
    vector<int> keys = hm_keys(buf);
    int pid = sample_one(keys);
    vector<record> recs = buf.at(pid);
    return sample_one(recs).input;
  };
}
