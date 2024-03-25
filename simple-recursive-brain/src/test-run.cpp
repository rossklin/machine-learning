#include <vector>
#include <random>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <omp.h>

#include "brain.hpp"
#include "util.hpp"

using namespace std;

enum tunnel_choice
{
    UP,
    DOWN,
    LEVEL,
    INVALID
};

struct GameState
{
    const static int viewport = 5, height = 5;
    vector<int> ceiling;
    vector<int> floor;
    vector<int> food;
    int pos; // Bot's position

    GameState() : ceiling(viewport, height), floor(viewport, 0), food(viewport, -1), pos(2) {}

    static int get_input_dim()
    {
        return (2 * viewport + 1) * height;
    }

    // True if a collision was fixed
    bool fix_collision()
    {
        if (pos >= ceiling[0])
        {
            // Collision
            pos = ceiling[0] - 1;
            return true;
        }
        else if (pos <= floor[0])
        {
            // Collision
            pos = floor[0] + 1;
            return true;
        }
        else
        {
            return false;
        }
    }

    float apply_choice(tunnel_choice choice)
    {
        if (choice == INVALID)
        {
            return -1;
        }

        if (choice == UP)
        {
            pos++;
        }
        else if (choice == DOWN)
        {
            pos--;
        }

        if (fix_collision())
        {
            return -1;
        }
        else if (food[0] == pos)
        {
            return 1;
        }
        else
        {
            return 0.5;
        }
    }

    // Update state and check collision, return true if collision was fixed
    bool shift()
    {
        copy(ceiling.begin() + 1, ceiling.end(), ceiling.begin());
        copy(floor.begin() + 1, floor.end(), floor.begin());
        copy(food.begin() + 1, food.end(), food.begin());

        // Ensure the tunnel is always traversable
        ceiling.back() = random_int(floor.back() + 2, height - 1);
        floor.back() = random_int(0, ceiling.back() - 2);

        // Add food in a traversable space
        food.back() = random_int(floor.back() + 1, ceiling.back() - 1);

        return fix_collision();
    }
};

const int GameState::viewport;
const int GameState::height;

tunnel_choice get_tunnel_choice(bool in1, bool in2)
{
    if (in1 && in2)
    {
        return INVALID;
    }
    else if (in1)
    {
        return UP;
    }
    else if (in2)
    {
        return DOWN;
    }
    else
    {
        return LEVEL;
    }
}

// d_in = 30, d_out = 2
float test_tunnel(Brain &b, GameState &state)
{

    // Add floor and ceiling to input
    vector<bool> input((2 * state.viewport + 1) * state.height, false);
    for (int col = 0; col < state.viewport; col++)
    {
        for (int row = 0; row < state.height; row++)
        {
            if (row >= state.ceiling[col] || row < state.floor[col])
            {
                input[col * state.height + row] = true;
            }
        }
    }

    // Add food to input
    for (int col = 0; col < state.viewport; col++)
    {
        if (state.food[col] >= 0)
        {
            input[state.viewport * state.height + col * state.height + state.food[col]] = true;
        }
    }

    // Add self to input
    input[2 * state.viewport * state.height + state.pos] = true;

    // Run network
    vector<float> output(2);
#ifndef NODEBUG
    cout << "Thinking";
#endif

    for (int i = 0; i < 10; i++)
    {
        b.set_input(input);
        b.update();
        vector<bool> buf = b.get_output();
        output[0] += buf[0];
        output[1] += buf[1];
#ifndef NODEBUG
        cout << ".";
#endif
    }

#ifndef NODEBUG
    cout << endl;
#endif

    tunnel_choice choice = get_tunnel_choice(output[0] > 3, output[1] > 3);

    // Update tunnel state
    bool collided = state.shift();
    float feedback = state.apply_choice(choice);
    if (collided)
    {
        feedback = -1;
    }

    return feedback;
}

/* TODO
It is probably difficult for the net to learn stuff when a node can not inhibit another node from firing
It would probably also be helpful if nodes could hold different amounts of energy in relation to the threshold, so it is possible for one node firing to trigger multiple other nodes at the same time
We also need a way to modify energy uptake in feedback
May also look into a method for crossing / GA
*/

// Usage: SRB n epochs threads
int main(int argc, char **argv)
{
    int d_in = GameState::get_input_dim(); // 55
    int d_out = 2, con_depth = 4;
    int n = atoi(argv[1]), epochs = atoi(argv[2]);
    int n_threads = argc == 4 ? atoi(argv[3]) : 6;
    int test_id_seed = 0;
    omp_set_num_threads(n_threads);

    if (n < 60)
    {
        cout << "This version requires at least 60 nodes!" << endl;
        exit(-1);
    }

#ifndef NODEBUG
    cout << "Starting!" << endl;
#endif

#ifdef NODEBUG
    cout << "thread_id, test_id, iteration, test, rolling_rate" << endl;
#endif

#pragma omp parallel
    while (true)
    {
        int test_id;
#pragma omp critical
        {
            test_id = test_id_seed++;
        }

        Brain b(n, 3, d_in, d_out, 40); // n, connectivity, d_in, d_out, track_l
        b.initialize(con_depth);
        float hit_rate = 0.125;
        int row = 1, col = 1;
        GameState state;

        for (int i = 1; i < epochs; i++)
        {
            const float r = test_tunnel(b, state);
            b.feedback(r);

            hit_rate = 0.999 * hit_rate + 0.001 * r;
#pragma omp critical
            {
#ifdef NODEBUG
                cout << omp_get_thread_num() << ", " << test_id << ", " << i << ", " << r << ", " << hit_rate << endl;
#else
                cout << "Row: " << omp_get_thread_num() << ", " << test_id << ", " << i << ", " << r << ", " << hit_rate << endl;
#endif
            }

            // // Sigmoid start at 0.143, hit ~0.255 at 1000, 0.356 at 2000, 0.63 at 10000
            // float min_level = 1 / (1 + exp(-i / (float)2200)) - 0.357;

            // if ((i % 200 == 0) && hit_rate < min_level)
            // {
            //     // Not performing well enough, start over
            //     break;
            // }
        }
    }

    return 0;
}