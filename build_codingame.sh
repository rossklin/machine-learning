#!/bin/bash
SOURCES="choice.cpp agent.cpp pod_agent.cpp game.cpp pod_game.cpp pod_game_generator.cpp game_generator.cpp utility.cpp evaluator.cpp tree_evaluator.cpp"
HEADERS="types.hpp utility.hpp agent.hpp pod_agent.hpp choice.hpp evaluator.hpp tree_evaluator.hpp game.hpp pod_game.hpp game_generator.hpp pod_game_generator.hpp"
FILES="$HEADERS $SOURCES"
BRAIN=$(cat $1)

echo $'void omp_dummy(int *x) {}\n' > pod_codingame.cpp

for f in $FILES; do cat $f >> pod_codingame.cpp; echo $'\n' >> pod_codingame.cpp; done

echo 'string agent_str = "'$BRAIN'";' >> pod_codingame.cpp

cat run_codingame.cpp >> pod_codingame.cpp

cat pod_codingame.cpp | sed '/#include ".*"$/d' | sed '/#pragma.*$/d' | sed 's/omp_.*(/omp_dummy(/g' | sed 's/omp_lock_t/int/g' > pod_codingame2.cpp

mv pod_codingame2.cpp pod_codingame.cpp