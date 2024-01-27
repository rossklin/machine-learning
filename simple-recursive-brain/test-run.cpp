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

// d_in = 30, d_out = 2
float test_tunnel(Brain &b)
{
    const int viewport = 5, height = 5;
    static vector<int> ceiling(viewport, height), floor(viewport, 0), food(viewport, -1);
    static int pos = 2;

    // Add floor and ceiling to input
    vector<bool> input((2 * viewport + 1) * height, false);
    for (int col = 0; col < viewport; col++)
    {
        for (int row = 0; row < height; row++)
        {
            if (row >= ceiling[col] || row < floor[col])
            {
                input[col * height + row] = true;
            }
        }
    }

    // Add food to input
    for (int col = 0; col < viewport; col++)
    {
        if (food[col] >= 0)
        {
            input[viewport * height + col * height + food[col]] = true;
        }
    }

    // Add self to input
    input[2 * viewport * height + pos] = true;

    // Run network
    b.set_input(input);
    b.update();
    vector<bool> output = b.get_output();

    // Analyze result
    bool valid_output = true;
    if (output[0] && output[1])
    {
        valid_output = false;
    }
    else if (output[0] && pos > 0)
    {
        pos--;
    }
    else if (output[1] && pos < height - 1)
    {
        pos++;
    }
    else if (output[0] || output[1])
    {
        valid_output = false;
    }
    bool ok_step = pos >= floor[0] && pos < ceiling[0];
    bool got_food = pos == food[0];
    float feedback = -1;
    if (ok_step && valid_output)
    {
        if (got_food)
        {
            feedback = 1;
        }
        else
        {
            feedback = 0.5;
        }
    }

    // Update tunnel state
    static vector<int> new_ceiling(viewport, 5), new_floor(viewport, 0), new_food(viewport, -1);
    copy(ceiling.begin() + 1, ceiling.end(), new_ceiling.begin());
    copy(floor.begin() + 1, floor.end(), new_floor.begin());
    copy(food.begin() + 1, food.end(), new_food.begin());

    // Find ceiling min that is not impossible
    int ceil_min = 0;
    for (int i = 0; i < viewport; i++)
    {
        // TODO
        ceil_min = max(ceil_min, floor[i] - (height - i + ...));
    }

    // TODO same for floor

    // new_ceiling.back() = random_int(floor.back() + 1, height);
    // new_floor.back() = random_int(0, ceiling.back() - 1);

    // TODO add food

    // TODO swap state

    return feedback;
}

// Test problem: step on 3x3 grid without hitting walls
float test_step(Brain &b, const int d_in, const int d_out)
{
    static int row = 1, col = 1;

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
}

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
            const float r = test_step(b, d_in, d_out);
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