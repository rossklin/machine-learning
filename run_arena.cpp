#include <cstring>

#include "arena.hpp"
#include "evaluator.hpp"
#include "pod_game.hpp"

using namespace std;

int main(int argc, char **argv) {
  int threads = 6;
  int ngames = 64;
  int ppt = 2;
  int tpg = 2;
  int tree_depth = 10;
  float preplim = 0.5;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "debug")) {
      threads = 1;
      tree_depth = 5;
      ngames = 4;
      preplim = 0.01;
    } else if (!strcmp(argv[i], "quick")) {
      threads = 5;
      tree_depth = 7;
      ngames = 10;
      preplim = 0.2;
    } else if (!strcmp(argv[i], "threads")) {
      threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "ngames")) {
      ngames = atoi(argv[++i]);
    }
  }

  game_ptr g(new pod_game(tpg, ppt, tree_depth));
  arena a(g, threads, ppt, tpg, preplim);
  a.evolution(ngames);

  return 0;
}
