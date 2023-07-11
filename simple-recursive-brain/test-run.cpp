#include <vector>
#include <random>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>

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

    set<int> fired_at_set()
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
    bool test_connectivity(pair<int, int> range1, pair<int, int> range2)
    {
        // Implementation of flood fill
        set<int> flooded, horizon;
        for (int i = range1.first; i <= range1.second; i++)
        {
            // Start with node i as the horizon
            bool is_connected = false;
            horizon.insert(i);
            while (horizon.size())
            {
                int j = *horizon.begin();
                if (j >= range2.first && j <= range2.second)
                {
                    // This node (i) is connected to range2!
                    is_connected = true;
                }
                horizon.erase(j);
                flooded.insert(j);

                for (auto x : edges[j])
                {
                    horizon.insert(x.first);
                }

                // Calculate the set difference manually because the "set_difference" function doesn't work for sets lol
                for (auto k : flooded)
                {
                    horizon.erase(k);
                }
            }
            if (!is_connected)
            {
                return false;
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
        for (int i = 0; i < n; i++)
        {
            float wsum = 0;

            for (auto x : edges[i])
            {
                wsum += x.second;
            }

            for (auto x : edges[i])
            {
                edges[i][x.first] = x.second / wsum;
            }
        }
    }

    // Select a random edge target that is not self and also not an input node
    int random_edge_target(int self)
    {
        int test_target;
        do
        {
            test_target = random_int(0, n - 1);
        } while (test_target == self || test_target < d_in);
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
    map<int, int> get_fired_parents_at_time(int j, int _t)
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
        const float chill_factor = 0.1;
        const float gamma = 0.5;
        const string prefix = string(time_of_output - time, '>') + " $ ";

        if (time_of_output - time > 10 || time < 1)
        {
            cout << prefix << "Reached end of time on node " << i << endl;
            return;
        }

        cout << prefix << "Testing " << i << " at time " << time << endl;

        const map<int, int> fired_parents = get_fired_parents_at_time(i, time);
        for (auto j : fired_parents)
        {
            // Sanity check
            if (j.second >= time)
            {
                throw logic_error("Parent fired at time geq current time!");
            }

            edges[j.first][i] = chill_factor * pow(gamma, time_of_output - time) * fabs(r) * sign + edges[j.first][i];
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
            cout << "Feedback 0, nothing to do!" << endl;
            return;
        }

        cout << "Giving feedback " << r << " at time " << t << endl;

        // Loop over output nodes

        for (int i = d_in; i < d_in + d_out; i++)
        {
            const bool output_did_fire = nodes[i].fired_at_set().contains(t);
            const int sign = (2 * output_did_fire - 1) * ((r > 0) - (r < 0)); // 1 = increase energy, -1 = reduce energy
            cout << "Starting recursive feedback for output node " << (i - d_in + 1) << " which " << (output_did_fire ? "did" : "did not") << " fire, giving sign " << sign << endl;
            feedback_recursively(i, r, sign, t, t, !output_did_fire);
        }

        cout << "Starting edge cleanup" << endl;

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
                    cout << "Removing edge from " << i << " to " << x.first << endl;
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
                cout << "Added edge from " << i << " to " << test_target << " with width " << edges[i][test_target] << endl;
            }
        }

        // After modifying edges, we need to normalize them again
        normalize_edges();
    }

    void initialize()
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
            cout << "Create edges: attempt " << attempts << endl;
        } while (!(test_connectivity(input_range, output_range)));

        for (int i = 0; i < n; i++)
        {
            nodes[i].energy_uptake = random_float(0, 0.1);
        }
    }

    vector<bool> get_output()
    {
        vector<bool> res(d_out);
        for (int i = d_in; i < d_in + d_out; i++)
        {
            res[i - d_in] = nodes[i].fired_at_set().contains(t);
        }
        return res;
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

int main(int argc, char **argv)
{
    Brain b(10, 3, 2, 2, 20);
    b.initialize();
    for (int i = 0; i < 5; i++)
    {
        // Test with random input
        for (int j = 0; j < b.d_in; j++)
        {
            b.nodes[j].energy = random_float(0, 2);
        }
        b.update();
        cout << "Step " << i << ": output: ";
        for (auto r : b.get_output())
        {
            {
                cout << r << ", ";
            }
        }
        cout << endl;
    }
    b.feedback(0.2);
    for (int i = 0; i < 5; i++)
    {
        // Test with random input
        for (int j = 0; j < b.d_in; j++)
        {
            b.nodes[j].energy = random_float(0, 2);
        }
        b.update();
        cout << "Step " << i << ": output: ";
        for (auto r : b.get_output())
        {
            {
                cout << r << ", ";
            }
        }
        cout << endl;
    }
    b.feedback(-0.2);
    cout << "Success!" << endl;
    return 0;
}