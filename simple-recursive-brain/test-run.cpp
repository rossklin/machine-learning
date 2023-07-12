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

struct Node
{
    // The state of the node is it's energy buffer and iterations passed since it last fired
    float energy;
    float energy_uptake;
    vector<int> fired_at;

    // Track parents to avoid work and hassle
    vector<int> parents;

    Node()
    {
        energy = 0;
        energy_uptake = 0;
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
    vector<map<int, float>> edges;

    // Return true if each node in range1 is connected to any node in range2 and each node in range2 is reachable from at least one node in range1
    bool test_connectivity(pair<int, int> range1, pair<int, int> range2, int max_depth) const
    {
        // Implementation of flood fill
        set<int> flooded, horizon;
        for (int i = range1.first; i <= range1.second; i++)
        {
            // Start with node i as the horizon
            bool is_connected = false;
            horizon.insert(i);
            int depth = 0;
            while (horizon.size() && depth++ < max_depth)
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
                return false; // Node i in range1 is not connected to range2
            }
        }

        // Verify that all of range2 was flooded
        for (int i = range2.first; i <= range2.second; i++)
        {
            if (!flooded.contains(i))
            {
                return false;
            }
        }

        return true;
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
                wsum += x.second;
            }
            x[i] = wsum;

            for (auto x : edges[i])
            {
                edges[i][x.first] = x.second / wsum;
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
            test_target = random_int(0, n - 1);
            if (count++ > 1e4)
            {
                throw runtime_error("Failed to select random edge target!");
            }
        } while (test_target == self || test_target < d_in || edges[self].contains(test_target));
        return test_target;
    }

    // Create initial edges, between one and connectivity edges per node, where whidth sums to 1.
    void create_edges(int connectivity)
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
            float wsum = 0;
            edges[i].clear();
            for (int j = 0; j < n_connect; j++)
            {
                int test_target = random_edge_target(i);
                const float w = random_float(0, 1);
                edges[i][test_target] = w;
                nodes[test_target].parents.push_back(i);
                wsum += w;
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

        // Increment time here, so the fired_at timestamps match the output state after the update
        t++;

        // Identify fired nodes, gather transmitted energy and reset energ on fired nodes
        for (int i = 0; i < n; i++)
        {
            if (nodes[i].energy >= 1)
            {
                // This node fired. Transmit energy through edges and set energy to zero.
                for (auto x : edges[i])
                {
                    energy_transmitted[x.first] += x.second;
                }
                node_buf[i].energy = 0;
                node_buf[i].fired_at.push_back(t);
            }
        }

        // Add transmitted energy and energy uptake to target nodes and clear old fired_at times
        for (int i = 0; i < n; i++)
        {
            node_buf[i].energy += energy_transmitted[i] + nodes[i].energy_uptake;
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

    void feedback_recursively(int i, float r, int sign, int time, int time_of_output, bool search_non_fired_ancestors)
    {
        const float chill_factor = 0.01;
        const float gamma = 0.5;
        const string prefix = string(time_of_output - time, '>') + " $ ";

        if (time_of_output - time > 10 || time < 1)
        {
#ifdef VERBOSE
            {
                cout << prefix << "Reached end of time on node " << i << endl;
            }
#endif
            return;
        }

        const map<int, int> fired_parents = get_fired_parents_at_time(i, time);

#ifdef VERBOSE
        {
            cout << prefix << "Testing " << i << " at time " << time << ": parents ";
            for (auto j : nodes[i].parents)
            {
                cout << j << "[" << (fired_parents.contains(j) ? "F" : "O") << "], ";
            }
            cout << endl;
        }
#endif

        for (auto j : fired_parents)
        {
            // Sanity check
            if (j.second >= time)
            {
                throw logic_error("Parent fired at time geq current time!");
            }

            const float amount = chill_factor * pow(gamma, time_of_output - time) * fabs(r) * sign;
            edges[j.first][i] = amount + edges[j.first][i];
#ifdef VERBOSE
            {

                cout << prefix << "Modified edge from " << j.first << " to " << i << " by " << amount << " to " << edges[j.first][i] << endl;
            }
#endif
            feedback_recursively(j.first, r, sign, j.second, time_of_output, false); // Always stop searching non-fired ancestors if there are fired parents
        }

        // If there were no fired parents and searching non-fired ancestors is allowed, continue to all parents with a single time step
        if (search_non_fired_ancestors && fired_parents.empty())
        {
            for (auto j : nodes[i].parents)
            {
                feedback_recursively(j, r, sign, time - 1, time_of_output, true);
            }
        }
    }

    // Give feedback
    // Positive:
    // For output nodes that fired, increase the width of edges from ancestors that did fire
    // For output nodes that did not fire, decrease the width of edges from ancestors that did fire
    // Negative:
    // For output nodes that fired, decrease the width of edges from ancestors that did fire
    // For output nodes that did not fire, increase the width of edges from ancestors that did fire (and intermediate?)
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

        // Loop over output nodes

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

            // Remove edges that are no longer positive
            const map<int, float> edge_buf = edges[i];
            for (auto x : edge_buf)
            {
                if (x.second <= 0)
                {
                    // This node (i) is no longer a parent of j
                    erase_if(nodes[x.first].parents, [i](int j)
                             { return j == i; });

                    edges[i].erase(x.first);
#ifdef VERBOSE
                    {

                        cout << "Removing edge from " << i << " to " << x.first << endl;
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
                const int test_target = random_edge_target(i);
                const float w = random_float(0, 1);
                edges[i][test_target] = w;
                nodes[test_target].parents.push_back(i);
#ifdef VERBOSE
                {

                    cout << "Added edge from " << i << " to " << test_target << " with width " << edges[i][test_target] << endl;
                }
#endif
            }
        }

        // After modifying edges, we need to normalize them again
        normalize_edges();
    }

    void initialize(int connection_depth)
    {
        pair<int, int> input_range = {0, d_in - 1};
        pair<int, int> output_range = {d_in, d_in + d_out - 1};
        int attempts = 0;
        do
        {
            create_edges(connectivity);
            if (attempts++ > 100)
            {
                throw runtime_error("100 attempts reached!");
            }
#ifdef VERBOSE
            {

                cout << "Create edges: attempt " << attempts << endl;
            }
#endif
        } while (!(test_connectivity(input_range, output_range, connection_depth)));

        for (int i = 0; i < n; i++)
        {
            nodes[i].energy_uptake = random_float(0, 0.1);
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

vector<bool> get_target(int d_in, int i)
{
    vector<bool> input(d_in, false);
    int target = (i / 10) % d_in;
    input[target] = true;
    return input;
}

int main(int argc, char **argv)
{
    int d_in = 2, d_out = 2, con_depth = 3;
    int n = atoi(argv[1]), epochs = atoi(argv[2]);
    int n_threads = 6;
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

        Brain b(n, 3, d_in, d_out, 20);
        b.initialize(con_depth);
        float hit_rate = 0.25;

        for (int i = 1; i < epochs; i++)
        {
            vector<bool> input = get_target(d_in, i);
            b.set_input(input);
            b.update();

            // Train at predicting input
            vector<bool> output = get_target(d_in, i + 1);
            bool test = true;
            for (int j = 0; j < d_in; j++)
            {
                test = test && (b.nodes[d_in + j].fired_at_set().contains(b.t) == output[j]);
            }
            b.feedback(2 * test - 1);

            hit_rate = 0.998 * hit_rate + 0.002 * test;
#pragma omp critical
            {
                cout << omp_get_thread_num() << ", " << test_id << ", " << i << ", " << test << ", " << hit_rate << endl;
            }

            if ((i % 200 == 0) && hit_rate < 0.25 + 0.00002 * i)
            {
                // Not performing well enough, start over
                break;
            }
        }
    }

    return 0;
}