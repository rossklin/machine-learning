#pragma once

#include <set>
#include <utility>
#include <functional>
#include <vector>
#include <map>

#include "node.hpp"
#include "edge.hpp"
#include "util.hpp"

struct Brain
{
    int n;
    int d_in, d_out;
    int connectivity;
    time_point t;
    int track_length;
    std::vector<Node> nodes;
    std::vector<std::map<node_index, Edge>> edges;
    float chill_factor_base;
    float gamma;
    float p_change_type;

    std::set<node_index> find_disconnected_nodes(std::pair<node_index, node_index> range1, std::pair<node_index, node_index> range2, int max_depth = 0) const;
    bool test_connectivity(std::pair<node_index, node_index> range1, std::pair<node_index, node_index> range2, int max_depth = 0) const;
    void normalize_edges();
    node_index random_edge_target(node_index self) const;
    void add_edge(node_index i, node_index j);
    node_index random_walk(std::pair<node_index, node_index> start_range, int steps, std::function<std::vector<node_index>(node_index)> option_selector) const;
    void create_edges(int connectivity, int connection_depth);
    void update();
    std::map<time_point, std::set<node_index>> get_fired_parents_at_time(node_index j, time_point _t) const;
    void feedback_recursively(node_index i, float r, int sign, time_point time, time_point time_of_output, bool search_non_fired_ancestors);
    void feedback(float r);
    void initialize(int connection_depth);
    std::vector<bool> get_output() const;
    void set_input(std::vector<bool> x);
    bool is_input_node(node_index idx) const;
    bool is_output_node(node_index idx) const;
    Brain(int _n, int _connectivity, int _d_in, int _d_out, int track_l);
};