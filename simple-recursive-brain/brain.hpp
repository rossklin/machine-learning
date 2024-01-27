#pragma once

#include <set>
#include <utility>
#include <functional>
#include <vector>
#include <map>

#include "node.hpp"
#include "edge.hpp"

struct Brain
{
    int n;
    int d_in, d_out;
    int connectivity;
    int t;
    int track_length;
    std::vector<Node> nodes;
    std::vector<std::map<int, Edge>> edges;
    float chill_factor_base;
    float gamma;
    float p_change_type;

    std::set<int> find_disconnected_nodes(std::pair<int, int> range1, std::pair<int, int> range2, int max_depth = 0) const;
    bool test_connectivity(std::pair<int, int> range1, std::pair<int, int> range2, int max_depth = 0) const;
    void normalize_edges();
    int random_edge_target(int self) const;
    void add_edge(int i, int j);
    int random_walk(std::pair<int, int> start_range, int steps, std::function<std::vector<int>(int)> option_selector) const;
    void create_edges(int connectivity, int connection_depth);
    void update();
    std::map<int, int> get_fired_parents_at_time(int j, int _t) const;
    void feedback_recursively(int i, float r, int sign, int time, int time_of_output, bool search_non_fired_ancestors);
    void feedback(float r);
    void initialize(int connection_depth);
    std::vector<bool> get_output() const;
    void set_input(std::vector<bool> x);
    Brain(int _n, int _connectivity, int _d_in, int _d_out, int track_l);
};