#include <set>
#include "brain.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "util.hpp"
#include <iostream>

using namespace std;

// Return nodes in range 1 that are not connected to range 2, plus nodes in range 2 that are not reachable from range 1
set<node_index> Brain::find_disconnected_nodes(pair<node_index, node_index> range1, pair<node_index, node_index> range2, int max_depth) const
{
    // Implementation of flood fill
    set<node_index> flooded, horizon, not_connected;
    for (node_index i = range1.first; i <= range1.second; i++)
    {
        // Start with node i as the horizon
        bool is_connected = false;
        horizon.insert(i);
        flooded.clear();
        int depth = 0;
        while (horizon.size() && (max_depth == 0 || depth++ < max_depth))
        {
            set<node_index> horizon_buf;
            for (auto j : horizon)
            {
                if (j >= range2.first && j <= range2.second)
                {
                    // This node (i) is connected to range2!
                    is_connected = true;
                }
                flooded.insert(j);

                for (auto x : edges[j])
                {
                    horizon_buf.insert(x.first);
                }

                // Calculate the set difference manually because the "set_difference" function doesn't work for sets lol
                for (auto k : flooded)
                {
                    horizon_buf.erase(k);
                }
            }
            swap(horizon, horizon_buf);
        }

        if (!is_connected)
        {
            not_connected.insert(i); // Node i in range1 is not connected to range2
        }
    }

    // Verify that all of range2 was flooded
    for (node_index i = range2.first; i <= range2.second; i++)
    {
        if (!flooded.contains(i))
        {
            not_connected.insert(i);
        }
    }

    return not_connected;
}

// Return true if the brain has no "disconnected" nodes
bool Brain::test_connectivity(pair<node_index, node_index> range1, pair<node_index, node_index> range2, int max_depth) const
{
    return find_disconnected_nodes(range1, range2, max_depth).empty();
}

// Normalize edge widths to sum to 1 for energy conservation
void Brain::normalize_edges()
{
    for (node_index i = 0; i < n; i++)
    {
        float wsum = 0;

        for (auto x : edges[i])
        {
            wsum += x.second.width;
        }

        for (auto x : edges[i])
        {
            edges[i][x.first].width = x.second.width / wsum;
        }
    }
}

// Select a random target for a new edge that is not self and also not an input node
node_index Brain::random_edge_target(node_index self) const
{
    int test_target;
    int count = 0;
    do
    {
        test_target = random_int(d_in, n - 1);
        if (count++ > 1e4)
        {
            throw runtime_error("Failed to select random edge target!");
        }
    } while (test_target == self || edges[self].contains(test_target));
    return test_target;
}

// Assume here that edges[i] exists, but edges[i][j] does not
void Brain::add_edge(node_index i, node_index j)
{
    if (edges[i].contains(j))
    {
        throw new runtime_error("Attempted to add edge that already exists!");
    }

    edges[i][j] = Edge();
    edges[i][j].width = random_float(0, 1);
    edges[i][j].is_inhibitor = random_float(0, 1) < 0.1;
    nodes[j].parents.push_back(i);
}

/* This function performs a random walk process.
It starts at a randomly selected node within a given range (start_range),
and it moves a given number of steps (steps).
The movement from node to node is driven by a function (option_selector) which selects the next node from a set of possible options.
If at any point there are no options for next move (i.e., the function has reached a dead-end), it restarts the walk.
The process is repeated until a successful path is found or the counter exceeds 100, in which case an exception is thrown.
The function returns the final node reached after a successful walk. */
node_index Brain::random_walk(pair<node_index, node_index> start_range, int steps, function<vector<node_index>(node_index)> option_selector) const
{
    bool success;
    int counter = 0;
    node_index target;
    do
    {
        if (counter++ > 100)
        {
            throw runtime_error("Failed to find valid connector path for disconnected input/output node after 100 tries!");
        }

        target = random_int(start_range.first, start_range.second);
        success = true;
        for (int j = 0; j < steps; j++)
        {
            vector<node_index> options = option_selector(target);

            if (options.empty())
            {
                success = false;
                break;
            }
            target = options[random_int(0, options.size() - 1)];
        }
    } while (!success);
    return target;
}

bool Brain::is_input_node(node_index idx) const
{
    return idx < d_in;
}

bool Brain::is_output_node(node_index idx) const
{
    return idx >= d_in && idx < d_in + d_out;
}

// Create initial edges, between one and connectivity edges per node, where whidth sums to 1.
// Guarantee all nodes are reachable from input in the end
// Guarantee output is reachable from input in connection_depth steps
void Brain::create_edges(int connectivity, int connection_depth)
{
    // In case of re-run, clear the parent index
    for (auto &n : nodes)
    {
        n.parents.clear();
    }

    for (node_index i = 0; i < n; i++)
    {
        // Do not create outgoing edges for output nodes
        if (is_output_node(i))
        {
            continue;
        }

        const int n_connect = random_int(1, connectivity);
        edges[i].clear();
        for (node_index j = 0; j < n_connect; j++)
        {
            add_edge(i, random_edge_target(i));
        }
    }

    // Connect all non-connected nodes to a random connected node
    set<node_index> not_connected = find_disconnected_nodes(make_pair(0, d_in - 1), make_pair(d_in, n - 1));

    // Sanity check - should not be possible!
    if (not_connected.size() >= n - d_in)
    {
        throw runtime_error("Create edges: all nodes were disconnected!");
    }

    for (auto i : not_connected)
    {
        // Each input node should have at least one edge to a non-input node
        if (is_input_node(i))
        {

            cout << "An input node " << i << " did not have an edge to a non-input node (total " << edges[i].size() << " edges)!" << endl;
            for (auto x : edges[i])
            {
                cout << x.first << ",";
            }
            cout << endl;
            throw runtime_error("No edge from input node!");
        }

        // Here we must have a non input node that was not reachable from an input node
        // Select a parent that is connected, that is not self, not an output node and not a child node
        node_index new_parent;
        do
        {
            new_parent = random_int(0, n - 1);
        } while (not_connected.contains(new_parent) || is_output_node(new_parent) || new_parent == i || edges[i].contains(new_parent));

        // Since new_parent was connected to the input, it can not already be our parent when we our selves are not connected
        add_edge(new_parent, i);
    }

    // Fix input nodes not connected to output nodes
    not_connected = find_disconnected_nodes(make_pair(0, d_in - 1), make_pair(d_in, d_in + d_out - 1), connection_depth);
    for (auto i : not_connected)
    {
        if (i < d_in)
        {
            // Input node that could not reach output

            // Random walk backwards from a random output - there likely exists at least one output node that has a backwards path that does not end prematurely at an input
            // A case which fails could be when all input nodes are connected to one output node and one input node connects to all output nodes
            node_index target = random_walk(make_pair(d_in, d_in + d_out - 1), connection_depth - 1, [this](node_index j)
                                            {
                    vector<node_index> options = nodes[j].parents;
                    erase_if(options, [this](node_index k)
                        { return is_input_node(k); });
                    return options; });

            // Connect to this node which has a known path to the output
            add_edge(i, target);
        }
        else
        {
            // Output node that was not reached by input
            // Random walk forwards from a random input without hitting an output
            node_index target = random_walk(make_pair(0, d_in - 1), connection_depth - 1, [this](node_index j)
                                            {
                    vector<node_index> options;
                    for (auto x : edges[j]) {
                        if (!is_output_node(x.first)) {
                            options.push_back(x.first);
                        }
                    }
                    return options; });

            // Connect to this node which has a known path from the input
            add_edge(target, i);
        }
    }

    normalize_edges();
}

// Run one increment of the simulation
// Or modify the firing so we don't loose energy
void Brain::update()
{
    // Update the state in node_buf based on the state in nodes, then swap them to get the new state in nodes
    vector<Node> node_buf = nodes;
    vector<float> energy_transmitted(n, 0);
    vector<float> inhibition_transmitted(n, 0);

    // Increment time here, so the fired_at timestamps match the output state after the update
    t++;

    // Identify fired nodes, gather transmitted energy and reset energy on fired nodes
    for (node_index i = 0; i < n; i++)
    {
        bool was_inhibited = nodes[i].inhibition >= 1;
        if (was_inhibited)
        {
            // Inhibition should activate and reset regardless of whether the node fires
            node_buf[i].inhibition = 0;
        }

        if (nodes[i].energy >= 1 && !was_inhibited)
        {
            // This node fired. Transmit energy through edges and set energy to zero.
            for (auto x : edges[i])
            {
                if (x.second.is_inhibitor)
                {
                    inhibition_transmitted[x.first] += nodes[i].firepower * x.second.width;
                }
                else
                {
                    energy_transmitted[x.first] += nodes[i].firepower * x.second.width;
                }
            }
            node_buf[i].energy = 0;
            node_buf[i].fired_at.push_back(t);
        }
    }

    // Add transmitted energy and energy uptake to target nodes and clear old fired_at times
    for (node_index i = 0; i < n; i++)
    {
        node_buf[i].energy += energy_transmitted[i] + nodes[i].energy_uptake;
        node_buf[i].inhibition += inhibition_transmitted[i];
        erase_if(node_buf[i].fired_at, [this](time_point time)
                 { return time < t - track_length; });
    }

    // Update state
    nodes.swap(node_buf);
}

typedef map<node_index, map<time_point, float>> FeedbackMap;

// Apply the feedback without normalizing the edges (for internal use)
void Brain::feedback_frontier(float r)
{
    // We want to reward the specific temporal pattern, ie that the output nodes fired at the times they did, in relation to the input pattern
    FeedbackMap frontier;
    time_point offset_time = t;
    const float not_fired_contribution_factor = 0.25;
    const float edge_adjustment_factor = 0.001;
    const float sadness_increment = 0.001;
    const float modification_tracker_increment = 0.001;

    if (r == 0)
    {
        return;
    }

    // Add output nodes to frontier
    for (node_index i = d_in; i < d_in + d_out; i++)
    {
        frontier[i][offset_time] = r;
    }

    while (!frontier.empty())
    {
        FeedbackMap frontier_buf;
        for (auto x : frontier)
        {
            const node_index i = x.first;
            for (auto y : x.second)
            {
                const float r2 = y.second;
                const time_point t2 = y.first;
                const time_point t1 = nodes[i].last_fired_before(t2); // Will be -1 if the node did not fire befor t2
                const bool did_fire = nodes[i].did_fire_at(t2);
                bool any_active_parents = false;

                for (node_index parent_idx : nodes[i].parents)
                {
                    bool parent_did_fire = false;
                    for (time_point parent_fired_time : nodes[parent_idx].fired_at)
                    {
                        const float gamma = pow(temporal_discount_factor, t - parent_fired_time);

                        // Set a limit when the feedback gets too small
                        if (fabs(gamma * r2) < 1e-6 || parent_fired_time < t1 || parent_fired_time >= t2)
                        {
                            continue;
                        }

                        parent_did_fire = true;
                        any_active_parents = true;

                        // Determine whether we like that the parent fired at us
                        // Used energy (Y/N) * Got energy (Y/N) * Happy with result (Y/N)
                        const int used_type_sign = (2 * did_fire - 1);
                        const int parent_contributed = used_type_sign * edges[parent_idx][i].get_type_sign();
                        const float modified_feedback = gamma * parent_contributed * r2;

                        // Adjust edges to parents: get more if we liked what they did, less otherwise
                        edges[parent_idx][i].width += edge_adjustment_factor * modified_feedback;

                        // Add parents to new frontier with sign adjusted feedback
                        frontier_buf[parent_idx][parent_fired_time] += modified_feedback;
                    }

                    if (!parent_did_fire)
                    {
                        // Give feedback for not firing
                        const float target_time = t2 - 1;
                        const float gamma = pow(temporal_discount_factor, t - target_time);

                        if (fabs(gamma * r2) < 1e-6)
                        {
                            // Determine whether we like that the parent did not fire at us (value these "contributions" relatively less)
                            const float parent_contributed = did_fire ? -1 : 1;
                            const float modified_feedback = gamma * not_fired_contribution_factor * parent_contributed * r2;

                            // Add parents to new frontier with sign adjusted feedback
                            frontier_buf[parent_idx][target_time] += modified_feedback;
                        }
                    }
                }

                // This value  tracks whether the node has mostly had positive or negative feedback lately
                // Typical amounts are in magnitude 1e-3 ish?
                const float feedback_sign = ((r2 > 0) - (r2 < 0));
                nodes[i].modification_tracker = modification_tracker_increment * feedback_sign + (1 - modification_tracker_increment) * nodes[i].modification_tracker;

                // Something like marking the node as sad if it gets negative feeback for not firing and didn't have any incoming energy
                const bool make_sad = r2 < 0 && !any_active_parents && !did_fire;
                if (make_sad)
                {
                    nodes[i].sadness = sadness_increment + (1 - sadness_increment) * nodes[i].sadness;
                }
            }
        }

        frontier = frontier_buf;

        offset_time--;
        if (offset_time >= t - track_length)
        {
            const float gamma = pow(temporal_discount_factor, t - offset_time);

            // Add output nodes to frontier with a time offset
            for (node_index i = d_in; i < d_in + d_out; i++)
            {
                frontier[i][offset_time] = gamma * r;
            }
        }
    };
}

// Give feedback

void Brain::feedback(float r)
{
    if (r == 0)
    {
#ifdef VERBOSE
        {

            cout << "Feedback 0, nothing to do!" << endl;
        }
#endif
        return;
    }

#ifdef VERBOSE
    {
        cout << "Giving feedback " << r << " at time " << t << endl;
    }
#endif

    // Apply the feedback, updating edges and trackers, without normalizing edges
    feedback_frontier(r);

#ifdef VERBOSE
    {

        cout << "Starting edge cleanup" << endl;
    }
#endif

    for (node_index i = 0; i < n; i++)
    {
        // Skip output nodes since they do not have outgoing edges
        if (i >= d_in && i < d_in + d_out)
        {
            continue;
        }

        // Switch type of edges that are no longer positive
        const map<node_index, Edge> edge_buf = edges[i];
        for (auto x : edge_buf)
        {
            if (x.second.width < 0)
            {
                edges[i][x.first].is_inhibitor = !edges[i][x.first].is_inhibitor;
                edges[i][x.first].width *= -1;

#ifdef VERBOSE
                {

                    cout << "Switch type due to negative width for edge " << i << " -> " << x.first << endl;
                }
#endif
            }
        }

        // Add a new edge with small probability if we are an active node with few edges
        const float rate_of_fire = nodes[i].fired_at.size() / (float)track_length;
        float p_add = 0.1 * (1 - edges[i].size() / (float)connectivity) * rate_of_fire;
        if (edges[i].empty())
        {
            p_add = 1;
        }

        if (random_float(0, 1) < p_add)
        {
            add_edge(i, random_edge_target(i));
#ifdef VERBOSE
            {

                cout << "Added edge from " << i << endl;
            }
#endif
        }

        // Change energy uptake and firepower based on modification tracker - positive means the node often is requested to grow it's edges so makes sense to grow the whole node, and vv
        // In fully one-sided feedback, the modification tracker will at most 0.005, realistically 0.01 * 0.8 * 0.5^3 = 1e-3
        // The signs are the same for inhibitor edges - positive tracker means we want it to inhibit more and vv
        const float track = nodes[i].modification_tracker;
        if (fabs(track) > 1e-4)
        {
            nodes[i].energy_uptake += random_float(0, 1) * track;
            nodes[i].firepower += random_float(0, 1) * track;

#ifdef VERBOSE
            cout << "Modification for node " << i << " based on tracker " << track << endl;
#endif
        }

#ifdef VERBOSE
        cout << "Mod tracker for " << i << " = " << track << endl;
#endif

        // TODO help inactive nodes by connecting them to an active node or shifting some energy uptake to them
    }

    // After modifying edges, we need to normalize them again
    normalize_edges();
}

void Brain::initialize(int connection_depth)
{
    pair<node_index, node_index> input_range = {0, d_in - 1};
    pair<node_index, node_index> output_range = {d_in, d_in + d_out - 1};
    int attempts = 0;

    // Require input connected to output in connection_depth steps, and connected to each other node in any number of steps
    bool failed;
    do
    {
        if (attempts++ > 1000)
        {
            throw runtime_error("1000 attempts reached!");
        }

        failed = false;
        try
        {
            create_edges(connectivity, connection_depth);
        }
        catch (...)
        {
            cout << "Create edges failed!" << endl;
            failed = true;
        }

#ifdef VERBOSE
        {

            cout << "Create edges: attempt " << attempts << endl;
        }
#endif
        // This is just a sanity check as create_edges should guarantee correct connections
    } while (!(failed || (test_connectivity(input_range, output_range, connection_depth) && test_connectivity(input_range, make_pair(d_in, n - 1)))));

    for (node_index i = 0; i < n; i++)
    {
        nodes[i].energy_uptake = random_float(0, 0.1);
        nodes[i].firepower = random_float(0.1, 2);
    }
}

// Called after update, checks which output nodes fired at the last time point
vector<bool> Brain::get_output() const
{
    vector<bool> res(d_out);
    for (node_index i = d_in; i < d_in + d_out; i++)
    {
        res[i - d_in] = nodes[i].fired_at.back() == t;
    }
    return res;
}

void Brain::set_input(vector<bool> x)
{
    for (node_index k = 0; k < d_in; k++)
    {
        nodes[k].energy = x[k];
    }
}

/* Idea
Recursive network where nodes buffer energy. When they accumulate 1 energy, they send energy via edges that also sum to 1.
Positive feedback causes wider connections to recently fired parents of recently fired nodes and vice versa.
Edges can dissapear after negative feedback, and new edges can be added randomly, possibly with a preference to starting from or ending at active nodes.
Maybe passive nodes can dissappear?
Output is calculated by running some number of iterations after a new input ..?
*/

Brain::Brain(int _n, int _connectivity, int _d_in, int _d_out, int track_l)
{
    if (_n < 2)
    {
        throw runtime_error("Must use at least two nodes!");
    }

    n = _n;
    d_in = _d_in;
    d_out = _d_out;
    track_length = track_l;
    connectivity = _connectivity;
    t = 0;
    chill_factor_base = pow(10, random_float(-3, -1));
    temporal_discount_factor = random_float(0.1, 0.9);
    p_change_type = pow(10, random_float(-2, 0));
    nodes.resize(n);
    edges.resize(n);
}
