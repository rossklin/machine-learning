#include "evaluator.hpp"
#include <sstream>

using namespace std;

string evaluator::serialize() const {
  stringstream ss;
  ss << dim << " " << stable;
  return ss.str();
}

void evaluator::deserialize(stringstream &ss) {
  ss >> dim >> stable;
}