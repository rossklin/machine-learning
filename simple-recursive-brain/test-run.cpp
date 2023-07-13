#include <vector>
#include <random>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <omp.h>

using namespace std;

default_random_engine &get_engine()
{
    static random_device rd; // a seed source for the random number engine
    static default_random_engine gen(rd());
    return gen;
}

int random_int(int a, int b)
{
    uniform_int_distribution<> distrib(a, b);
    return distrib(get_engine());
}

float random_float(float a, float b)
{
    uniform_real_distribution<> distrib(a, b);
    return distrib(get_engine());
}

struct Edge
{
    float width;
    bool is_inhibitor;

    int get_type_sign() const
    {
        return is_inhibitor ? -1 : 1;
    }
};

struct Node
{
    // The state of the node is it's energy buffer and iterations passed since it last fired
    float energy;
    float inhibition;
    float energy_uptake;
    float firepower;
    float modification_tracker;
    vector<int> fired_at;

    // Track parents to avoid work and hassle
    vector<int> parents;

    Node()
    {
        energy = 0;
        inhibition = 0;
        energy_uptake = 0;
        firepower = 0;
        modification_tracker = 0;
    }

    set<int> fired_at_set() const
    {
        return set<int>(fired_at.begin(), fired_at.end());
    }
};

struct Brain
{
    int n;
    int d_in, d_out;
    int connectivity;
    int t;
    int track_length;
    vector<Node> nodes;
    vector<map<int, Edge>> edges;

    // Return true if each node in range1 is connected to any node in range2 and each node in range2 is reachable from at least one node in range1
    set<int> find_disconnected_nodes(pair<int, int> range1, pair<int, int> range2, int max_depth = 0) const
    {
        // Implementation of flood fill
        set<int> flooded, horizon, not_connected;
        for (int i = range1.first; i <= range1.second; i++)
        {
            // Start with node i as the horizon
            bool is_connected = false;
            horizon.insert(i);
            flooded.clear();
            int depth = 0;
            while (horizon.size() && (max_depth == 0 || depth++ < max_depth))
            {
                set<int> horizon_buf;
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
        for (int i = range2.first; i <= range2.second; i++)
        {
            if (!flooded.contains(i))
            {
                not_connected.insert(i);
            }
        }

        return not_connected;
    }

    bool test_connectivity(pair<int, int> range1, pair<int, int> range2, int max_depth = 0) const
    {
        return find_disconnected_nodes(range1, range2, max_depth).empty();
    }

    // Normalize edge widths to sum to 1 for energy conservation
    void normalize_edges()
    {
        vector<float> x(n);
        for (int i = 0; i < n; i++)
        {
            float wsum = 0;

            for (auto x : edges[i])
            {
                wsum += x.second.width;
            }
            x[i] = wsum;

            for (auto x : edges[i])
            {
                edges[i][x.first].width = x.second.width / wsum;
            }
        }

// Debug
#ifdef VERBOSE
        {
            x.erase(x.begin() + d_in, x.begin() + d_in + d_out);
            sort(x.begin(), x.end());
            cout << "Normalizing: lowest = " << x.front() << ", highest = " << x.back() << endl;
        }
#endif
    }

    // Select a random new edge target that is not self and also not an input node
    int random_edge_target(int self) const
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

    void add_edge(int i, int j)
    {
        edges[i][j].width = random_float(0, 1);
        edges[i][j].is_inhibitor = random_float(0, 1) < 0.1;
        nodes[j].parents.push_back(i);
    }

    int random_walk(pair<int, int> start_range, int steps, function<vector<int>(int)> option_selector) const
    {
        bool success;
        int counter = 0;
        int target;
        do
        {
            if (counter++ > 100)
            {
                throw runtime_error("Failed to find valid connector path for disconnected input/output node!");
            }

            target = random_int(start_range.first, start_range.second);
            success = true;
            for (int j = 0; j < steps; j++)
            {
                vector<int> options = option_selector(target);

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

    // Create initial edges, between one and connectivity edges per node, where whidth sums to 1.
    // Guarantee all nodes are reachable from input in the end
    void create_edges(int connectivity, int connection_depth)
    {
        // In case of re-run, clear the parent index
        for (auto &n : nodes)
        {
            n.parents.clear();
        }

        for (int i = 0; i < n; i++)
        {
            // Do not create outgoing edges for output nodes
            if (i >= d_in && i < d_in + d_out)
            {
                continue;
            }

            const int n_connect = random_int(1, connectivity);
            edges[i].clear();
            for (int j = 0; j < n_connect; j++)
            {
                add_edge(i, random_edge_target(i));
            }
        }

        // Connect all non-connected nodes to a random connected node
        set<int> not_connected = find_disconnected_nodes(make_pair(0, d_in - 1), make_pair(d_in, n - 1));

        // Sanity check - should not be possible!
        if (not_connected.size() >= n - d_in)
        {
            throw runtime_error("Create edges: all nodes were disconnected!");
        }

        for (auto i : not_connected)
        {
            // Each input node should have at least one edge to a non-input node
            if (i < d_in)
            {

                cout << "An input node " << i << " did not have an edge to a non-input node (total " << edges[i].size() << " edges)!" << endl;
                for (auto x : edges[i])
                {
                    cout << x.first << ",";
                }
                cout << endl;
                throw runtime_error("No edge from input node!");
            }

            // Select a parent that is connected, that is not self, not an output node and not a child node
            int new_parent;
            do
            {
                new_parent = random_int(0, n - 1);
            } while (not_connected.contains(new_parent) || (new_parent >= d_in && new_parent < d_in + d_out) || new_parent == i || edges[i].contains(new_parent));

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
                int target = random_walk(make_pair(d_in, d_in + d_out - 1), connection_depth - 1, [this](int j)
                                         {
                    vector<int> options = nodes[j].parents;
                    erase_if(options, [this](int k)
                        { return k < d_in; });
                    return options; });

                // Connect to this node which has a known path to the output
                add_edge(i, target);
            }
            else
            {
                // Output node that was not reached by input
                // Random walk forwards from a random input without hitting an output
                int target = random_walk(make_pair(0, d_in - 1), connection_depth - 1, [this](int j)
                                         {
                    vector<int> options;
                    for (auto x : edges[j]) {
                        if (x.first < d_in || x.first >= d_in + d_out) {
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
    void update()
    {
        vector<Node> node_buf = nodes;
        vector<float> energy_transmitted(n, 0);
        vector<float> inhibition_transmitted(n, 0);

        // Increment time here, so the fired_at timestamps match the output state after the update
        t++;

        // Identify fired nodes, gather transmitted energy and reset energ on fired nodes
        for (int i = 0; i < n; i++)
        {
            if (nodes[i].energy >= 1)
            {
                if (nodes[i].inhibition >= 1)
                {
                    // This node was inhibited from firing
                    node_buf[i].inhibition = 0;
                    continue;
                }

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
        for (int i = 0; i < n; i++)
        {
            node_buf[i].energy += energy_transmitted[i] + nodes[i].energy_uptake;
            node_buf[i].inhibition += inhibition_transmitted[i];
            erase_if(nodes[i].fired_at, [this](int time)
                     { return time < t - track_length; });
        }

        // Update state
        nodes.swap(node_buf);
    }

    // Calculate which parents of node j fired (and when) during the time period between the last time node j fired before _t and _t
    // These are the parents which contributed energy to the next time node j fired after time _t
    map<int, int> get_fired_parents_at_time(int j, int _t) const
    {
        map<int, int> fired_parents;
        vector<int> possible_times; // anti-chronological order
        for (int test = _t - 1; test > _t - 6; test--)
        {
            possible_times.push_back(test);
            if (nodes[j].fired_at_set().contains(test))
            {
                break;
            }
        }

        for (auto k : nodes[j].parents)
        {
            // Check if the node did fire at the correct time
            for (auto test : possible_times)
            {
                if (nodes[k].fired_at_set().contains(test))
                {
                    // Since possible_times are in anti-chronological order, this is guaranteed to be the last time the node fired in the window
                    fired_parents[k] = test;
                    break;
                }
            }
        }

        return fired_parents;
    }

    // Positive sign means we want to be more likely to fire and vv
    void feedback_recursively(int i, float r, int sign, int time, int time_of_output, bool search_non_fired_ancestors)
    {

        // Be less volatile as you get older
        // f(0) = 1
        // f(10000) = 0.1 => 1e8/h = -log(0.1) => h = -1e8 / log(0.1)
        // f(x) = e^(-x² / h)
        const float chill_factor = 0.01 * exp(-pow(t, 2) / (1e8 / -log(0.1)));
        const float gamma = 0.5;
        const string prefix = string(time_of_output - time, '>') + " $ ";

        if (time_of_output - time > 10 || time < 1)
        {
#ifdef VERBOSE_REC
            {
                cout << prefix << "Reached end of time on node " << i << endl;
            }
#endif
            return;
        }

        const map<int, int> fired_parents = get_fired_parents_at_time(i, time);

#ifdef VERBOSE_REC
        {
            cout << prefix << "Testing " << i << " at time " << time << ": parents ";
            for (auto j : nodes[i].parents)
            {
                cout << j << "[" << (fired_parents.contains(j) ? "F" : "O") << "], ";
            }
            cout << endl;
        }
#endif

        const float amount = chill_factor * pow(gamma, time_of_output - time) * fabs(r) * sign;
        for (auto j : fired_parents)
        {
            // Sanity check
            if (j.second >= time)
            {
                throw logic_error("Parent fired at time geq current time!");
            }

            const float track = nodes[j.first].modification_tracker;

            // Allow changing type instead of modifying width if type sign is opposite of amount and tracker supports amount
            int type_sign = edges[j.first][i].get_type_sign();
            const bool allow_type_change = type_sign * amount < 0 && track * amount > 0;

            // With small probability, switch type instead of changing width
            if (allow_type_change && random_float(0, 1) < 0.1 * fabs(track))
            {
                edges[j.first][i].is_inhibitor = !edges[j.first][i].is_inhibitor;
                type_sign *= -1;
#ifdef VERBOSE_REC
                {

                    cout << prefix << "Modified type of edge from " << j.first << " to " << i << " into " << (-1 * type_sign) << endl;
                }
#endif
            }
            else
            {
                // Reverse amount change if edge is inhibitor
                edges[j.first][i].width = type_sign * amount + edges[j.first][i].width;

#ifdef VERBOSE_REC
                {

                    cout << prefix << "Modified edge (t = " << type_sign << ") from " << j.first << " to " << i << " by " << (type_sign * amount) << " to " << edges[j.first][i].width << endl;
                }
#endif
            }

            // This value  tracks whether the node has mostly had positive or negative feedback lately
            // Typical amounts are in magnitude 1e-3 ish?
            nodes[j.first].modification_tracker = 0.999 * track + 0.001 * type_sign * amount;

            // If the edge is an inhibitor, we want the opposite effect for the parent
            feedback_recursively(j.first, r, type_sign * sign, j.second, time_of_output, false); // Always stop searching non-fired ancestors if there are fired parents
        }

        // If there were no fired parents and searching non-fired ancestors is allowed, continue to all parents with a single time step
        if (search_non_fired_ancestors && fired_parents.empty())
        {
            for (auto j : nodes[i].parents)
            {
                // If the edge is an inhibitor, we want the opposite effect for the parent
                feedback_recursively(j, r, edges[j][i].get_type_sign() * sign, time - 1, time_of_output, true);
            }
        }
    }

    // Give feedback

    void feedback(float r)
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

        for (int i = d_in; i < d_in + d_out; i++)
        {
            const bool output_did_fire = nodes[i].fired_at_set().contains(t);
            const int sign = (2 * output_did_fire - 1) * ((r > 0) - (r < 0)); // 1 = increase energy, -1 = reduce energy
#ifdef VERBOSE
            {

                cout << "Starting recursive feedback for output node " << (i - d_in + 1) << " which " << (output_did_fire ? "did" : "did not") << " fire, giving sign " << sign << endl;
            }
#endif
            feedback_recursively(i, r, sign, t, t, !output_did_fire);
        }

#ifdef VERBOSE
        {

            cout << "Starting edge cleanup" << endl;
        }
#endif

        for (int i = 0; i < n; i++)
        {
            // Skip output nodes since they do not have outgoing edges
            if (i >= d_in && i < d_in + d_out)
            {
                continue;
            }

            // Switch type of edges that are no longer positive
            const map<int, Edge> edge_buf = edges[i];
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

    void initialize(int connection_depth)
    {
        pair<int, int> input_range = {0, d_in - 1};
        pair<int, int> output_range = {d_in, d_in + d_out - 1};
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

        for (int i = 0; i < n; i++)
        {
            nodes[i].energy_uptake = random_float(0, 0.1);
            nodes[i].firepower = random_float(0.1, 2);
        }
    }

    vector<bool> get_output() const
    {
        vector<bool> res(d_out);
        for (int i = d_in; i < d_in + d_out; i++)
        {
            res[i - d_in] = nodes[i].fired_at_set().contains(t);
        }
        return res;
    }

    void set_input(vector<bool> x)
    {
        for (int k = 0; k < d_in; k++)
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

    Brain(int _n, int _connectivity, int _d_in, int _d_out, int track_l)
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
        nodes.resize(n);
        edges.resize(n);
    }
};

// vector<bool> get_target(int d_in, int i)
// {
//     vector<bool> input(d_in, false);
//     int target = (i / 10) % d_in;
//     input[target] = true;
//     return input;
// }

/* TODO
It is probably difficult for the net to learn stuff when a node can not inhibit another node from firing
It would probably also be helpful if nodes could hold different amounts of energy in relation to the threshold, so it is possible for one node firing to trigger multiple other nodes at the same time
We also need a way to modify energy uptake in feedback
May also look into a method for crossing / GA
*/

int main(int argc, char **argv)
{
    int d_in = 6, d_out = 4, con_depth = 4;
    int n = atoi(argv[1]), epochs = atoi(argv[2]);
    int n_threads = argc == 4 ? atoi(argv[3]) : 6;
    int test_id_seed = 0;
    omp_set_num_threads(n_threads);

    cout << "thread_id, test_id, iteration, test, rolling_rate" << endl;
#pragma omp parallel
    while (true)
    {
        int test_id;
#pragma omp critical
        {
            test_id = test_id_seed++;
        }

        Brain b(n, 3, d_in, d_out, 40);
        b.initialize(con_depth);
        float hit_rate = 0.125;
        int row = 1, col = 1;

        for (int i = 1; i < epochs; i++)
        {
            vector<bool> input(d_in, false);
            input[row] = true;
            input[3 + col] = true;
            b.set_input(input);
            b.update();

            // Train at predicting input
            vector<bool> output = b.get_output();
            int count = 0;
            int idx = -1;
            for (int j = 0; j < d_out; j++)
            {
                if (output[j])
                {
                    count++;
                    idx = j;
                }
            }

            float r = -1;

            if (count == 1)
            {
                r = -0.5;
                switch (idx)
                {
                case 0:
                    if (row > 0)
                    {
                        row--;
                        r = 1;
                    }
                    else
                    {
                        r = -1;
                    }
                    break;
                case 1:
                    if (col < 2)
                    {
                        col++;
                        r = 1;
                    }
                    else
                    {
                        r = -1;
                    }
                    break;
                case 2:
                    if (row < 2)
                    {
                        row++;
                        r = 1;
                    }
                    else
                    {
                        r = -1;
                    }
                    break;
                case 3:
                    if (col > 0)
                    {
                        col--;
                        r = 1;
                    }
                    else
                    {
                        r = -1;
                    }
                    break;
                default:
                    throw logic_error("Invalid output index!");
                }
            }

            b.feedback(r);

            hit_rate = 0.999 * hit_rate + 0.001 * (r == 1);
#pragma omp critical
            {
                cout << omp_get_thread_num() << ", " << test_id << ", " << i << ", " << r << ", " << hit_rate << endl;
            }

            // Sigmoid start at 0.143, hit ~0.255 at 1000, 0.356 at 2000, 0.63 at 10000
            float min_level = 1 / (1 + exp(-i / (float)2200)) - 0.357;

            if ((i % 200 == 0) && hit_rate < min_level)
            {
                // Not performing well enough, start over
                break;
            }
        }
    }

    return 0;
}