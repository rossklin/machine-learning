#include <vector>
#include <random>
#include <map>

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
    int time_since_fired;

    // Track parents to avoid work and hassle
    vector<int> parents;

    Node()
    {
        energy = 0;
        time_since_fired = 0;
    }
};

struct Brain
{
    int n;
    vector<Node> nodes;
    vector<map<int, float>> edges;

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
    void update()
    {
        vector<Node> node_buf = nodes;
        vector<float> energy_transmitted(n, 0);

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
                node_buf[i].time_since_fired = 0;
            }
        }

        // Add transmitted energy to target nodes and increment time
        for (int i = 0; i < n; i++)
        {
            node_buf[i].energy += energy_transmitted[i];
            node_buf[i].time_since_fired++;
        }

        // Update state
        nodes.swap(node_buf);
    }

    // Give feedback
    void feedback(float r)
    {
        const float chill_factor = 0.1;
        const float gamma = 0.5;

        // Apply feedback to edges
        for (int i = 0; i < n; i++)
        {
            // Don't go to far back in time
            if (nodes[i].time_since_fired > 4)
            {
                continue;
            }

            // We associate this node to the feedback based on how recently it fired
            float association_factor = pow(gamma, nodes[i].time_since_fired);

            // We associate the parents of this node to it's being fired based on how recently they fired before that
            for (auto j : nodes[i].parents)
            {
                const float gamma2 = 0.5;
                const int t_diff = nodes[j].time_since_fired - nodes[i].time_since_fired;

                // The parent only affected the current node if it fired at an earlier time
                if (t_diff < 1)
                {
                    continue;
                }

                // Don't go to for back in time here either
                if (t_diff > 4)
                {
                    continue;
                }

                const float parent_association = pow(gamma2, t_diff);

                // We should safely assume that if the parents of i contain j then the edges of j contain i
                // We increase or decrease the width of edges pointing to this node based on the feedback
                edges[j][i] += chill_factor * association_factor * parent_association * r;
            }
        }

        // Remove no width edges and re-normalize
        for (int i = 0; i < n; i++)
        {
            int edges_lost = 0;
            auto buf = edges[i];
            for (auto x : edges[i])
            {
                if (x.second <= 0)
                {
                    // We (node i) are no longer a parent of the targeted node (x.first)
                    erase_if(nodes[x.first].parents, [i](const int &j)
                             { return i == j; });
                    buf.erase(x.first);
                    edges_lost++;
                }
            }
            edges[i] = buf;

            // Compensate for edges lost somehow?

            // Re-normalize the edges
            float wsum = 0;
            for (auto y : edges[i])
            {
                wsum += y.second;
            }
            for (auto y : edges[i])
            {
                edges[i][y.second] /= wsum;
            }
        }
    }

    /* Idea
    Recursive network where nodes buffer energy. When they accumulate 1 energy, they send energy via edges that also sum to 1.
    Positive feedback causes wider connections to recently fired parents of recently fired nodes and vice versa.
    Edges can dissapear after negative feedback, and new edges can be added randomly, possibly with a preference to starting from or ending at active nodes.
    Maybe passive nodes can dissappear?
    Output is calculated by running some number of iterations after a new input ..?
    */

    Brain(int _n, int connectivity)
    {
        if (n < 2)
        {
            throw "Must use at least 2 nodes!";
        }

        n = _n;
        nodes.resize(n);
        edges.resize(n);
        create_edges(connectivity);
    }
};

int main(int argc, char **argv)
{
}