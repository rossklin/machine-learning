#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "game_generator.hpp"
#include "pod_game.hpp"
#include "pod_game_generator.hpp"
#include "population_manager.hpp"
#include "simple_pod_evaluator.hpp"
#include "team_evaluator.hpp"
#include "tree_evaluator.hpp"
#include "utility.hpp"

using namespace std;

agent_ptr agent_gen(int ppt, int cdim) {
  agent_ptr a(new pod_agent);
  // vector<evaluator_ptr> evals;
  // for (int i = 0; i < ppt; i++) evals.push_back(tree_evaluator::ptr(new tree_evaluator));
  // a->eval = team_evaluator::ptr(new team_evaluator(evals, 4));

  // Debug: run refbot structure with random weights
  tree_evaluator::ptr e = tree_evaluator::ptr(new tree_evaluator);
  e->example_setup(cdim);
  // int n = e->get_weights().size();
  // e->set_weights(vec_replicate<double>(bind(&rnorm, 0, 1), n));
  a->eval = e;
  a->label = "tree-pod-agent";
  a->learning_rate = 0;
  a->step_limit = 0.1;
  return a;
}

agent_ptr refbot_gen() {
  agent_ptr a(new pod_agent);
  a->eval = evaluator_ptr(new simple_pod_evaluator);
  a->label = "simple-pod-agent";
  return a;
}

const int SUPERVISION = 1;
const int REINFORCEMENT = 2;

void pure_train(int n) {
  unsigned int run_id = rand_int(0, INT32_MAX);
  cout << "Pure train: start run " << run_id << endl;
  omp_set_num_threads(6);

  int ppt = 1;
  pod_game_generator ggen(2, ppt, refbot_gen);
  vector<agent_ptr> pop(n);
  input_sampler isam = ggen.generate_input_sampler(10);
  input_sampler isam2 = ggen.generate_input_sampler(2);
  int cdim = ggen.choice_dim();
  set<int> ireq = ggen.required_inputs();
  // Debug
  ireq = {0, 1, 9, 10};
  vector<int> training_types(n, 0);

  for (int i = 0; i < n; i++) {
    if (u01() < 0.33) {
      training_types[i] = SUPERVISION;
    } else if (u01() < 0.5) {
      training_types[i] = REINFORCEMENT;
    } else {
      training_types[i] = SUPERVISION | REINFORCEMENT;
    }
  }

  // simulated data to train output to 0
  // Note: could train on refbot values instead (supervision)
  cout << "Generating init data..." << endl;
  vector<record> recs0 = vec_replicate(isam, 300);
  for (auto& r : recs0) {
    r.sum_future_rewards = r.opts[r.selected_option].output;  // simplest supervision data
    // for (auto& o : r.opts) o.output = rnorm(0, 0.01);
    // r.opts[r.selected_option].output = r.sum_future_rewards;
  }

  auto vgen = [ggen, ppt, cdim, isam, ireq, recs0, isam2]() -> agent_ptr {
    agent_ptr a = agent_gen(ppt, cdim);
    // a->initialize_from_input(isam, cdim, ireq);
    // a->eval->add_inputs(set_difference(ireq, a->eval->list_inputs()));

    double rc = 0;
    int n = 0;
    double n_pred;
    int ndata = recs0.size() * recs0.front().opts.size();
    double limit = ndata * 0.03;
    optim_result<double> res;
    int muts_used = 0;
    tree_evaluator::ptr ep = static_pointer_cast<tree_evaluator>(a->eval);
    int nw = ep->get_weights().size();

    cout << "BEFORE" << endl;
    cout << "Evaluator parameters:" << endl;
    cout << ep->root->printout() << endl;

    // now play a sample game
    game_ptr g = ggen.team_bots_vs(a);
    g->play(1);
    int clid = g->team_clone_ids(0).front();
    int rfid = g->team_clone_ids(1).front();

    cout << "Sample game speed: " << g->score_simple(clid) << endl;
    cout << "Compare refbot speed: " << g->score_simple(rfid) << endl;

    cout << "Select starting point..." << endl;
    vec w(nw, 0);
    do {
      ep->set_weights(w = vec_replicate<double>(bind(&rnorm, 0, 2), nw));
    } while (l2norm(ep->fgrad(w, recs0, a)) < 1);

    // TODO: seems constant 0 is always a tempting local optimum...
    cout << "Start updating" << endl;
    for (int i = 0; i < 100; i++) {
      vec x0 = ep->get_weights();
      res = a->eval->mod_update(recs0, a, rc);
      if (!res.success) {
        if (a->step_limit > 1e-3) {
          // Try again, slower
          a->step_limit = 0.3 * a->step_limit;
          ep->set_weights(x0);
          cout << "Slower" << endl;
        } else {
          break;
        }
      } else if (res.improvement < 1e-3 || res.obj <= limit) {
        break;
      }
    }

    // look at difference to refbot on some sample inputs
    agent_ptr ref = refbot_gen();

    cout << "Evaluator parameters:" << endl;
    cout << ep->root->printout() << endl;
    for (int i = 0; i < 1; i++) {
      cout << "Sample record comparison" << endl;
      record r = sample_one(recs0);
      sort(r.opts.begin(), r.opts.end(), [ref](option a, option b) { return a.output > b.output; });

      vec errs(r.opts.size());
      for (int i = 0; i < errs.size(); i++) {
        option o = r.opts[i];
        errs[i] = fabs(o.output - a->eval->evaluate(o.input));
        cout << o.output << comma << ref->eval->evaluate(o.input) << comma << a->eval->evaluate(o.input) << endl;
      }

      sort(errs.begin(), errs.end());
      cout << "Error q90: " << errs[0.9 * errs.size()] << endl;
      cout << "Error q95: " << errs[0.95 * errs.size()] << endl;
      cout << "Error max: " << errs.back() << endl;
    }

    // now play a sample game
    g = ggen.team_bots_vs(a);
    g->play(1);
    clid = g->team_clone_ids(0).front();
    rfid = g->team_clone_ids(1).front();

    cout << "AFTER" << endl;
    cout << "Sample game speed: " << g->score_simple(clid) << endl;
    cout << "Compare refbot speed: " << g->score_simple(rfid) << endl;

    if (res.obj <= limit) {
      return a;
    } else {
      return NULL;
    }
  };

  int agents_accepted = 0;
  int agents_discarded = 0;

  cout << "Pure train: generating initial pop" << endl;
  int counter = 0;

  // #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    do {
      cout << "Attempt for agent " << i << " of " << n << endl;
      pop[i] = vgen();
      if (pop[i]) {
        agents_accepted++;
      } else {
        agents_discarded++;
      }
    } while (!pop[i]);

    counter++;
    cout << ((100 * counter) / n) << "%  \r" << flush;
  }
  cout << endl;

  cout << "Accepted " << agents_accepted << " of " << (agents_accepted + agents_discarded) << " init agents." << endl;

  omp_lock_t writelock;
  omp_init_lock(&writelock);

  string fname = "data/pure-train-run-" + to_string(run_id) + ".csv";
  int batch_size;

  for (int epoch = 1; pop.size() > 1; epoch++) {
    batch_size = ceil(sqrt(epoch));
    cout << "Pure train: epoch " << epoch << ": batch size " << batch_size << ", " << pop.size() << " agents remaining" << endl;

    stringstream ss;

    counter = 0;
#pragma omp parallel for
    for (int i = 0; i < pop.size(); i++) {
      agent_ptr a = pop[i];
      if (!a->eval->stable) continue;
      a->set_exploration_rate(0.5 - 0.4 * psigmoid(a->score_simple.value_ma - 1, 0.3));

      double win_buf = 0;
      double speed_buf = 0;
      vector<vector<record>> training_data;

      // run batch_size games
      for (int k = 0; k < batch_size; k++) {
        game_ptr g = ggen.team_bots_vs(a);

        auto res = g->play(epoch);
        auto clone_ids = g->team_clone_ids(0);
        auto op_ids = g->team_clone_ids(1);

        if (training_types[i] & SUPERVISION) {
          for (auto pid : op_ids) training_data.push_back(res[pid]);
        }

        if (training_types[i] & REINFORCEMENT) {
          for (auto pid : clone_ids) training_data.push_back(res[pid]);
        }

        bool win = g->winner == a->team;
        bool tie = g->winner == -1;

        win_buf += win;

        int clone_id = g->team_clone_ids(a->team).front();
        speed_buf += g->score_simple(clone_id);
      }

      a->score_refbot.push(win_buf / batch_size);
      a->score_simple.push(speed_buf / batch_size);

      a->train(training_data, isam);

      omp_set_lock(&writelock);
      int sup = (training_types[i] & SUPERVISION) > 0;
      int rfm = (training_types[i] & REINFORCEMENT) > 0;
      ss << epoch << comma << sup << comma << rfm << comma << a->status_report() << endl;
      omp_unset_lock(&writelock);

      counter++;
      cout << ((100 * counter) / pop.size()) << "% done\r" << flush;
    }

    cout << endl;

    cout << "Writing stats and cleaning up" << endl;

    ofstream fmeta(fname, ios::app);
    fmeta << ss.str();
    fmeta.close();

    for (int i = 0; i < pop.size(); i++) {
      if (!pop[i]->eval->stable || (pop[i]->tstats.rate_successfull < 0.1 && pop[i]->age > 2)) {
        pop.erase(pop.begin() + i);
        training_types.erase(training_types.begin() + i);
        i--;
      }
    }
  }

  omp_destroy_lock(&writelock);
}

int main(int argc, char** argv) {
  int n = 100;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "n")) {
      n = atoi(argv[++i]);
    }
  }

  pure_train(n);
  return 0;
}
