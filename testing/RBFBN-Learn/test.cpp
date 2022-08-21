#include <omp.h>

#include <chrono>
#include <iostream>
#include <map>
#include <nlopt.hpp>
#include <random>
#include <vector>

#include "flann/flann.hpp"
#include "flann/io/hdf5.h"

using namespace std;

typedef string id_t;
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
  GENERATED_AT = 0,
  OUTPUT = 1,
  FEEDBACK = 2,
  NUM_TRACKING = 3
};

// Use the first d dims in index for input state, next two dims are output and feedback, they will be excluded from distance measure
// This is because point indices change when data is added to the index, so we can't connect indices to extra data externally
template <int d>
class Node {
  typedef flann::Index<flann::L2_SimpleLimited<data_t, d>> index_t;
  vector<id_t> inputs;
  index_t index;
  flann::SearchParams params;

 public:
  Node() : params(1024) {
  }

  data_t sample_output(vector<data_t> input) {
    const int nn = 10;
    const int n_prior = 10;
    const float bw2 = 0.1;
    const data_t x_prior = runif();
    const int n_data = index.size();

    // Make a query with right dimension but ignoring the tracking dimensions
    const vector<data_t> query_data(d + NUM_TRACKING, 0);
    copy(input.begin(), input.end(), query_data.begin());
    const flann::Matrix<data_t> query(&input[0], 1, d);

    vector<vector<int>> indices;
    vector<vector<float>> squared_dists;
    index.knnSearch(query, indices, squared_dists, nn, params);

    if (indices.empty() || indices[0].empty()) {
      return x_prior;
    }

    // Evalute a suggested output based on neighbours' historic feedback; estimate the feedback this output would give using a gaussian kernel on KNN
    const auto evalfun = [this, indices, squared_dists](const vector<double> &x, vector<double> &grad, void *f_data) -> double {
      const float suggested_output = x[0];
      float res = 0;
      float wsum = 0;
      float dres = 0;
      float dwsum = 0;
      for (int i = 0; i < indices[0].size(); i++) {
        const int idx = indices[0][i];
        auto p = index.getPoint(idx);
        const data_output = p[d + OUTPUT];
        const data_feedback = p[d + FEEDBACK];
        const input_d2 = squared_dists[0][i];
        const output_d2 = pow(suggested_output - data_output, 2);
        const total_d2 = input_d2 + output_d2;  // Consider these directions "perpendicular" (we may need a weighting if eg inputs are not normalized)
        const w = exp(-total_d2 / bw2);
        const inner_deriv = (-1 / bw2) * (2 * suggested_output - 2 * data_output);
        res += w * data_feedback;
        wsum += w;
        dres += data_feedback * w * inner_deriv;
        dwsum += w * inner_deriv;
      }
      grad[0] = dres * wsum - res * dwsum * pow(wsum, 2);  // (f/g)' = f' g - f g' g²
      return res / wsum;
    };

    nlopt::opt opt(nlopt::NLOPT_LD_LBFGS, dim);
    opt.set_max_objective(evalfun, NULL);
    opt.set_maxeval(20);
    opt.set_lower_bounds(vector<double>(1, 0));
    opt.set_upper_bounds(vector<double>(1, 1));
    vector<double> x = {0.5};
    double fopt = 0;

    try {
      nlopt::result result = opt.optimize(x, fopt);
    } catch (std::exception &e) {
      cout << "nlopt failed: " << e.what() << endl;
      res.success = false;
      res.obj = y;
      res.improvement = 0;
    }
  }
};
