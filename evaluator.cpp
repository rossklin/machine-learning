#include "evaluator.hpp"

#include <sstream>

#include "team_evaluator.hpp"
#include "tree_evaluator.hpp"
#include "types.hpp"
#include "utility.hpp"

using namespace std;

string evaluator::serialize() const {
  stringstream ss;
  ss << dim << sep << stable << sep << learning_rate;
  return ss.str();
}

void evaluator::deserialize(stringstream &ss) {
  ss >> dim >> stable >> learning_rate;
}

void evaluator::set_learning_rate(double r) { learning_rate = r; }

evaluator_ptr deserialize_evaluator(stringstream &ss) {
  evaluator_ptr eval;
  string tag;
  ss >> tag;

  if (tag == "tree") {
    eval = evaluator_ptr(new tree_evaluator);
  } else if (tag == "team") {
    eval = evaluator_ptr(new team_evaluator);
  } else {
    throw runtime_error("Invalid evaluator tag: " + tag);
  }

  eval->deserialize(ss);
  return eval;
}

string serialize_evaluator(evaluator_ptr e) {
  stringstream ss;
  ss << e->tag << sep << e->serialize();
  return ss.str();
}