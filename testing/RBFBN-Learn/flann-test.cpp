#include "flann/flann.hpp"

#include <omp.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "flann/io/hdf5.h"

using namespace std;
const int d = 4;

typedef flann::Index<flann::L2_SimpleLimited<float>> index_t;

float testval() {
  static default_random_engine gen;
  static normal_distribution<float> dist(0, 1);
  return dist(gen);
}

void extend_index(index_t &index, int n2 = 1000000) {
  vector<float> data2(n2 * d);
  generate(data2.begin(), data2.end(), testval);
  flann::Matrix<float> dataset2(&data2[0], n2, d);

  auto t1 = chrono::high_resolution_clock::now();
  index.addPoints(dataset2);
  auto t2 = chrono::high_resolution_clock::now();
  auto d1 = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
  cout << "Time to extend from " << (index.size() - n2) << " to " << index.size() << ": " << d1 << endl;
}

int main(int argc, char **argv) {
  int n = 100000;
  int nn = 30;

  auto t0 = chrono::high_resolution_clock::now();

  vector<float> data(n * d);
  generate(data.begin(), data.end(), testval);
  flann::Matrix<float> dataset(&data[0], n, d);

  int nq = 3;
  vector<float> qdata(nq * d);
  generate(qdata.begin(), qdata.end(), testval);
  flann::Matrix<float> query(&qdata[0], nq, d);

  vector<vector<int>> indices;
  vector<vector<float>> dists;  // construct an randomized kd-tree index using 4 kd-trees
  index_t index(dataset, flann::KDTreeIndexParams(4), flann::L2_SimpleLimited<float>(2));
  auto params = flann::SearchParams(1024);
  params.cores = argc > 1 ? stoi(argv[1]) : 1;

  auto t1 = chrono::high_resolution_clock::now();
  index.buildIndex();
  auto t2 = chrono::high_resolution_clock::now();

  index.knnSearch(query, indices, dists, nn, params);
  auto t3 = chrono::high_resolution_clock::now();

  const int idx = indices[0][0];
  const int idx2 = indices[0][1];
  cout << "Qdata " << qdata[0] << "x" << qdata[1] << endl;
  cout << "Query point" << endl;
  cout << query[0][0] << "x" << query[0][1] << endl;
  for (int i = 0; i < d; i++) {
    cout << query[0][i] << ",";
  }
  cout << endl;
  cout << "Nearest 5" << endl;
  cout << dataset[idx][0] << "x" << dataset[idx][1] << endl;
  for (int j = 0; j < 5; j++) {
    cout << "Dist = " << dists[0][j] << ": ";
    auto p = index.getPoint(indices[0][j]);
    for (int i = 0; i < d; i++) {
      cout << p[i] << ",";
    }
    cout << endl;
  }
  // cout << "Second" << endl;
  // cout << dataset[idx2][0] << "x" << dataset[idx2][1] << endl;

  // cout << "Start" << endl;
  // for (auto row : indices) {
  //   cout << "Row: ";
  //   for (auto idx : row) {
  //     cout << idx << ", ";
  //   }
  //   cout << endl;
  // }
  // cout << "end" << endl;

  // auto d0 = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
  // auto d1 = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
  // auto d2 = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();

  // cout << "Startup time: " << d0 << endl;
  // cout << "Time to build index: " << d1 << endl;  // Does not seem to use openmp, ~2s with -O2, ~5s without for 1e6 datapoints
  // cout << "Time to search: " << d2 << endl;       // Got ~2x speedup on 6 cores for 1e6 points, 1.8 resp 3.5 ms

  // auto p1 = index.getPoint(idx);
  // auto p2 = index.getPoint(idx2);

  // cout << "Pre update p1: " << p1[0] << "x" << p1[1] << endl;
  // cout << "Pre update p2: " << p2[0] << "x" << p2[1] << endl;

  // for (int i = 0; i < 3; i++) {
  //   extend_index(index);
  // }

  // p1 = index.getPoint(idx);
  // p2 = index.getPoint(idx2);

  // cout << "Post update p1: " << p1[0] << "x" << p1[1] << endl;
  // cout << "Post update p2: " << p2[0] << "x" << p2[1] << endl;

  return 0;
}
