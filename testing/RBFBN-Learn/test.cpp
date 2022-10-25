#include <omp.h>

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <nlopt.hpp>
#include <random>
#include <vector>

#include "flann/flann.hpp"
#include "flann/io/hdf5.h"

using namespace std;

typedef float data_t;

// See how well we can learn a simple elementary component function
float test_target_d3(vector<data_t> input) {
  return sin(input[0] - 1) - pow(input[1], 2) * cos(input[2]);
}

float runif() {
  static default_random_engine gen;
  static uniform_real_distribution<data_t> dist(0, 1);
  return dist(gen);
}

enum Tracking {
  OUTPUT = 0,
  FEEDBACK = 1,
  GENERATED_AT = 2,
  NUM_TRACKING = 3
};

class Node;

struct evalfun_params {
  Node *node;
  vector<vector<int>> indices;
  vector<vector<float>> squared_dists;
};

double node_evalfun(unsigned int n, const double *x, double *grad, void *f_data);

// Use the first d dims in index for input state, next two dims are output and feedback, they will be excluded from distance measure
// This is because point indices change when data is added to the index, so we can't connect indices to extra data externally
class Node {
 public:
  typedef flann::L2_SimpleLimited<data_t> dist_t;
  typedef flann::Index<dist_t> index_t;
  flann::SearchParams params;
  int d_input;
  shared_ptr<index_t> index;
  vector<vector<data_t>> buffer;

  Node(int d_in) : params(1024), d_input(d_in) {}

  data_t sample_output(vector<data_t> input) {
    assert(input.size() == d_input);

    const int nn = 10;
    const int n_prior = 10;
    const data_t x_prior = runif();

    if (!index) {
      input.push_back(x_prior);
      buffer.push_back(input);
      return x_prior;
    }

    const int n_data = index->size();

    // Make a query with right dimension but ignoring the tracking dimensions
    const int d = d_input + NUM_TRACKING;
    vector<data_t> query_data(d, 0);
    copy(input.begin(), input.end(), query_data.begin());
    const flann::Matrix<data_t> query(&query_data[0], 1, d);

    evalfun_params ep;
    ep.node = this;
    index->knnSearch(query, ep.indices, ep.squared_dists, nn, params);

    cout << "Found indices:" << endl;
    for (auto idx : ep.indices[0]) {
      data_t *p = index->getPoint(idx);
      cout << idx << ": ";
      for (int i = 0; i < d; i++) cout << p[i] << ",";
      cout << endl;
    }
    cout << endl;

    nlopt::opt opt(nlopt::algorithm::LD_LBFGS, 1);
    opt.set_max_objective(node_evalfun, static_cast<void *>(&ep));
    opt.set_maxeval(20);
    opt.set_lower_bounds(vector<double>(1, 0));
    opt.set_upper_bounds(vector<double>(1, 1));
    vector<double> x = {x_prior};
    double fopt = 0;

    try {
      nlopt::result result = opt.optimize(x, fopt);
    } catch (std::exception &e) {
      cout << "nlopt failed: " << e.what() << endl;
      x[0] = runif();
      fopt = -1;
    }

    // Store input + output in buffer
    input.push_back(x[0]);
    buffer.push_back(input);
    return x[0];
  }

  void give_feedback(data_t feedback) {
    if (buffer.empty()) {
      cout << "Warning: feedback with no data in buffer!" << endl;
      return;
    }

    const int d = d_input + NUM_TRACKING;
    vector<data_t> x;
    x.reserve(d * buffer.size());

    for (auto sample : buffer) {
      sample.resize(d);
      sample[d_input + FEEDBACK] = feedback;
      sample[d_input + GENERATED_AT] = 0;  // TODO
      x.insert(x.end(), sample.begin(), sample.end());
    }
    flann::Matrix<data_t> data(&x[0], buffer.size(), d);
    if (index) {
      cout << "Adding points to index" << endl;
      index->addPoints(data);
    } else {
      const auto idx_params = flann::KDTreeIndexParams(4);
      cout << "Constructing index" << endl;
      index = make_shared<index_t>(index_t(data, idx_params, dist_t(d_input)));
      cout << "Building index" << endl;
      index->buildIndex();
    }
  }
};

// Estimate the feedback we will get for selecting a given output in this situation
double node_evalfun(unsigned int n, const double *x, double *grad, void *f_data) {
  const auto ep = static_cast<evalfun_params *>(f_data);
  const float suggested_output = x[0];
  const float bw2 = 0.1;
  float res = 0;
  float wsum = 0;
  float dres = 0;
  float dwsum = 0;
  for (int i = 0; i < ep->indices[0].size(); i++) {
    const int idx = ep->indices[0][i];
    auto p = ep->node->index->getPoint(idx);
    const data_t data_output = p[ep->node->d_input + OUTPUT];
    const data_t data_feedback = p[ep->node->d_input + FEEDBACK];
    const data_t input_d2 = ep->squared_dists[0][i];
    const data_t output_d2 = pow(suggested_output - data_output, 2);
    const data_t total_d2 = input_d2 + output_d2;  // Consider these directions "perpendicular" (we may need a weighting if eg inputs are not normalized)
    const data_t w = exp(-total_d2 / bw2);
    const data_t inner_deriv = (-1 / bw2) * (2 * suggested_output - 2 * data_output);
    res += w * data_feedback;
    wsum += w;
    dres += data_feedback * w * inner_deriv;
    dwsum += w * inner_deriv;
  }
  grad[0] = dres * wsum - res * dwsum * pow(wsum, 2);  // (f/g)' = f' g - f g' g²

  // Add penalty term for x outside [0,1]
  if (x[0] > 1) {
    res -= pow(x[0] - 1, 2);
    grad[0] -= 2 * x[0] - 2;
  } else if (x[0] < 0) {
    res -= pow(x[0], 2);
    grad[0] -= 2 * x[0];
  }

  if (wsum < 1e-6) {
    return 0;
  } else {
    return res / wsum;
  }
}

// TODO At loop 1/0 "Found indices" are sensible except index "0" which has extreme values, then nlopt fails, then at loop 1/1 all indices have extreme values, one value is nan
// Note no feedback between, can this indicate flann::Matrix constructor expects external party to manage data pointer?
int main() {
  Node n(3);
  vector<data_t> input(3);
  cout << "Starting" << endl;

  for (int i = 0; i < 10; i++) {
    data_t feedback = 0;
    for (int j = 0; j < 10; j++) {
      cout << "Loop" << i << "/" << j << endl;
      generate(input.begin(), input.end(), runif);
      const data_t target = test_target_d3(input);
      cout << "Generating guess" << endl;
      const data_t guess = n.sample_output(input);
      feedback += exp(-pow((guess - target) / 0.3, 2));
    }
    n.give_feedback(feedback / 10);
    cout << "Iteration " << i << "feedback = " << feedback << endl;
  }
}