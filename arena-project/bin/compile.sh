#!/bin/bash

# function definitions
cat ml.cpp ml_pod.cpp | sed 's/\(#include "ml.*cpp"\)/\/\/\1/g' > pod_codingame.cpp

# brain definitions
echo 'string p0="'$(cat $1)'";' >> pod_codingame.cpp
echo 'string p1="'$(cat $1)'";' >> pod_codingame.cpp

# codingame turn function
cat run_codingame.cpp >> pod_codingame.cpp
