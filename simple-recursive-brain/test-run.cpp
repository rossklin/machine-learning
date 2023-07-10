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
    set<int> fired_at;

    // Track parents to avoid work and hassle
    vector<int> parents;

    Node()
    {
        energy = 0;
    }
};

struct Brain
{
    int n;
    int d_in, d_out;
    int connectivity;
    int t;
    vector<Node> nodes;
    vector<map<int, float>> edges;

    // Return true if each node in range1 is connected to any node in range2
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
                    break;
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

        return true;
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
            const int n_connect = random_int(1, connectivity);
            float wsum = 0;
            edges[i].clear();
            for (int j = 0; j < n_connect; j++)
            {
                int test_target = random_int(0, n - 1);
                while (test_target == i)
                {
                    test_target = random_int(0, n - 1);
                }
                const float w = random_float(0, 1);
                edges[i][test_target] = w;
                nodes[test_target].parents.push_back(i);
                wsum += w;
            }

            // Normalize edge widths to sum to 1 for energy conservation
            for (auto x : edges[i])
            {
                edges[i][x.first] = x.second / wsum;
            }
        }
    }

    // Run one increment of the simulation
    // TODO We may need to add and/or leak some energy for energy stability
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
                node_buf[i].fired_at.insert(t);
            }
        }

        // Add transmitted energy to target nodes and increment time
        for (int i = 0; i < n; i++)
        {
            node_buf[i].energy += energy_transmitted[i];
        }

        // Update state
        nodes.swap(node_buf);
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

        const float chill_factor = 0.1;
        const float gamma = 0.5;
        // Loop over output nodes

        for (int i = d_in; i < d_out; i++)
        {
            const bool output_did_fire = nodes[i].fired_at.contains(t);
            const int sign = (2 * output_did_fire - 1) * ((r > 0) - (r < 0)); // 1 = increase energy, -1 = reduce energy
            set<int> horizon = {i};
            set<int> flooded;
            int depth = 0;

            // Check that I know how loop scopes work
            if (flooded.size() > 0)
            {
                throw runtime_error("You don't know how loop scopes work!");
            }

            // BFS so we can track what level we're at and scale the "blame" (association)
            while (horizon.size() > 0 && depth < 5)
            {
                set<int> horizon_buf;
                depth++;
                for (auto j : horizon)
                {
                    flooded.insert(j);
                    // Calculate which time points could have affected the firing of node j
                    set<int> possible_times;
                    for (int test = t - depth; test > t - depth - 5; t--)
                    {
                        possible_times.insert(test);
                        if (nodes[j].fired_at.contains(test))
                        {
                            break;
                        }
                    }

                    // Process edges to node j, then add parents to horizon
                    for (auto k : nodes[j].parents)
                    {
                        // Check if the node did fire at the correct time
                        bool fired_at_relevant_time = false;
                        for (auto test : possible_times)
                        {
                            if (nodes[k].fired_at.contains(test))
                            {
                                fired_at_relevant_time = true;
                                break;
                            }
                        }

                        // Not sure if this condition is needed, maybe just update all ancestor edges regardless?
                        if (fired_at_relevant_time)
                        {
                            // Process edge from parent k to node j: make it send more/less energy based on sign
                            edges[j][k] = chill_factor * pow(gamma, depth) * fabs(r) * sign + edges[j][k];
                        }

                        // Add parents to horizon regardless to continue searching for an ancestor that did fire
                        horizon_buf.insert(nodes[k].parents.begin(), nodes[k].parents.end());
                    }
                }

                // Update horizon
                for (auto j : flooded)
                {
                    horizon_buf.erase(j);
                }
                horizon.swap(horizon_buf);
            }
        }

        // TODO add/remove edges

        // TODO renormalize edges
    }

    void initialize()
    {
        pair<int, int> range1 = {0, d_in - 1};
        pair<int, int> range2 = {d_in, d_in + d_out - 1};
        int attempts = 0;
        do
        {
            create_edges(connectivity);
            if (attempts++ > 100)
            {
                throw runtime_error("100 attempts reached!");
            }
            cout << "Create edges: attempt " << attempts << endl;
        } while (!(test_connectivity(range1, range2) && test_connectivity(range2, range1)));
    }

    /* Idea
    Recursive network where nodes buffer energy. When they accumulate 1 energy, they send energy via edges that also sum to 1.
    Positive feedback causes wider connections to recently fired parents of recently fired nodes and vice versa.
    Edges can dissapear after negative feedback, and new edges can be added randomly, possibly with a preference to starting from or ending at active nodes.
    Maybe passive nodes can dissappear?
    Output is calculated by running some number of iterations after a new input ..?
    */

    Brain(int _n, int _connectivity, int _d_in, int _d_out)
    {
        if (_n < 2)
        {
            throw runtime_error("Must use at least two nodes!");
        }

        n = _n;
        d_in = _d_in;
        d_out = _d_out;
        connectivity = _connectivity;
        nodes.resize(n);
        edges.resize(n);
    }
};

int main(int argc, char **argv)
{
    Brain b(10, 3, 2, 2);
    b.initialize();
    cout << "Success!" << endl;
    return 0;
}