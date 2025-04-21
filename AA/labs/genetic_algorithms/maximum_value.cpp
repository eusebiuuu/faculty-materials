#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;
ifstream fin("input.in");
ofstream fout("output.out");

int const M = 100, P = 20;
long long const INF = 1e18;

int a, b, c;
int left_bound, right_bound, encoding_length, precision;
double crossover_probability;
double gene_mutation_probability;
long long closest_pw_2 = 1;

void find_closest_power_of_2(int l, int r) {
    long long pw10 = 1;
    for (int i = 1; i <= precision; ++i) {
        pw10 *= 10;
    }

    long long intervals_count = 1LL * (r - l) * pw10;
    
    while (closest_pw_2 < intervals_count) {
        closest_pw_2 <<= 1LL;
        encoding_length++;
    }
}

bitset<M> encode(int l, int r, double num) {
    // step = (b - a) / 2 ^ exp
    long long interval_idx = floor(1.0 * (num - l) * closest_pw_2 / (r - l));

    bitset<M> binary_representation = interval_idx;
    return binary_representation;
}

double decode(int l, int r, bitset<M> bin_repr) {
    long long interval_idx = bin_repr.to_ullong();

    double step = 1.0 * (r - l) / closest_pw_2;
    double interval_start = 1.0 * l + 1.0 * interval_idx * step;

    return interval_start;
}

double generate_random(double start, double end) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double num = dis(gen);
    int diff = end - start;
    double actual_num = start + diff * num;
    return actual_num;
}

double get_fitness(double v) {
    return 1.0 * a * v * v + b * v + c;
}

void print_encoding(bitset<M> encoding) {
    fout << encoding.to_string().substr(M - encoding_length);
}

int get_idx(vector<double> intervals, double num) {
    int l = 0, r = (int) intervals.size() - 1;

    while (l < r) {
        int mid = (l + r) / 2;
        if (intervals[mid] < num) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    return l;
}

vector<bitset<M>> selection(vector<bitset<M>> population, int population_count, bool verbose) {
    double min_val = 0.0, max_value = -INF;
    int pos_max = 0;

    vector<pair<double, int>> fitness_values;
    for (int i = 0; i < population_count; ++i) {
        double real_value = decode(left_bound, right_bound, population[i]);
        double current_fitness = get_fitness(real_value);
        fitness_values.push_back({current_fitness, i});

        min_val = min(min_val, current_fitness);

        if (max_value < current_fitness) {
            pos_max = i;
            max_value = current_fitness;
        }
    }

    vector<bitset<M>> selected_population;
    selected_population.push_back(population[pos_max]);
    if (verbose) {
        fout << "Cromozomul " << pos_max + 1 << " a fost extras prin selectia elitista\n";
    }
    fitness_values.erase(fitness_values.begin() + pos_max);

    double sum = 0.0;

    for (auto &elem : fitness_values) {
        elem.first += min_val;
        sum += elem.first;
    }

    if (verbose) {
        fout << "\nProbabilitati selectie:\n";
    }

    int sz = fitness_values.size();
    vector<double> intervals = {0};
    for (int i = 0; i < sz; ++i) {
        double proportion = 1.0 * fitness_values[i].first / sum;
        if (verbose) {
            fout << "Cromozomul " << i + 1 << ": ";
            fout << setprecision(P) << proportion << '\n';
        }
        double last_sum = intervals.back();
        intervals.push_back(last_sum + proportion);
    }

    if (verbose) {
        for (double x : intervals) {
            fout << setprecision(P) << x << ' ';
        }
        fout << "\nSelectia proportionala:\n";
    }

    for (int i = 0; i < sz; ++i) {
        double num = generate_random(0.0, 1.0);
        int idx = get_idx(intervals, num);
        if (verbose) {
            fout << "Am generat numarul " << setprecision(P) << num;
            fout << ", deci vom alege cromozomul " << idx << '\n';
        }

        selected_population.push_back(population[fitness_values[idx - 1].second]);
    }

    return selected_population;
}

void crossover(vector<bitset<M>> &encodings, vector<int> idx, int cnt, int p1, int p2, bool verbose) {
    if (verbose) {
        fout << "S-au incrucisat cromozomii ";
        for (int i = 0; i < cnt; ++i) {
            fout << ", " << i;
        }
        fout << ":\n";
        for (int i = 0; i < cnt; ++i) {
            print_encoding(encodings[idx[i]]);
            fout << ' ';
        }
        fout << "\nBreak points: " << p1 << ' ' << p2 << '\n';
    }

    bitset<M> aux = encodings[idx.back()];
    for (int i = cnt - 1; i > 0; --i) {
        for (int j = encoding_length - p2; j < encoding_length - p1; ++j) {
            encodings[idx[i]][j] = encodings[idx[i - 1]][j];
        }
    }
    for (int j = encoding_length - p2; j < encoding_length - p1; ++j) {
        encodings[idx[0]][j] = aux[j];
    }

    if (verbose) {
        fout << "Rezultat: ";
        for (int i = 0; i < cnt; ++i) {
            print_encoding(encodings[idx[i]]);
            fout << ' ';
        }
        fout << '\n';
    }
}

void apply_crossover(vector<bitset<M>> &population, int sz, double prob, bool verbose) {
    if (verbose) {
        fout << "Probabilitatea de incrucisare este " << prob << '\n';
    }
    vector<int> marked_positions;
    for (int i = 0; i < sz; ++i) {
        double num = generate_random(0.0, 1.0);
        
        if (verbose) {
            fout << i + 2 << ": ";
            print_encoding(population[i]);
            fout << ", p = " << num;
            if (num <= prob) {
                fout << " <= " << prob << ", deci participa\n";
            } else {
                fout << " > " << prob << ", deci nu participa\n";
            }
        }

        if (num <= prob) {
            marked_positions.push_back(i);
        }
    }

    if (verbose) {
        fout << "Recombinari:\n";
    }
    
    int pos_count = (int) marked_positions.size();
    for (int i = 0; i < pos_count && pos_count > 1; i += 2) {
        double break_point1 = generate_random(0, encoding_length - 1);
        double break_point2 = generate_random(0, encoding_length - 1);
        int pos1 = floor(break_point1), pos2 = floor(break_point2);

        if (i == pos_count - 3) {
            vector<int> curr_pos(marked_positions.begin() + i, marked_positions.end());
            crossover(population, curr_pos, 3, min(pos1, pos2), max(pos1, pos2), verbose);
            break;
        }

        vector<int> curr_pos(marked_positions.begin() + i, marked_positions.begin() + i + 2);
        crossover(population, curr_pos, 2, min(pos1, pos2), max(pos1, pos2), verbose);
    }
}

void apply_mutations(vector<bitset<M>> &population, int sz, double prob, bool verbose) {
    if (verbose) {
        fout << "Probabilitatea de mutatie este " << prob << '\n';
        fout << "Au fost modificati cromozomii:\n";
    }

    bitset<M> mask;
    int idx = 1;

    for (bitset<M> &encoding : population) {
        bool changed = false;
        for (int j = 0; j < encoding_length; ++j) {
            double num = generate_random(0.0, 1.0);
            mask[j] = num <= prob;
            changed |= num <= prob;
        }

        if (changed && verbose) {
            fout << idx << '\n';
        }

        encoding ^= mask;
        idx++;
    }
}

void print_max(vector<bitset<M>> population, int sz, int step) {
    double max_num = get_fitness(decode(left_bound, right_bound, population[0]));
    double sum = 0.0;
    
    for (auto encoding : population) {
        double curr_fitness = get_fitness(decode(left_bound, right_bound, encoding));
        max_num = max(curr_fitness, max_num);
        sum += curr_fitness;
    }

    fout << "Max fitness " << step << ": " << setprecision(P) << max_num << '\n';
    fout << "Mean fitness " << step << ": " << setprecision(P) << sum / sz << '\n';
}

int main() {
    int population_count;
    int steps;

    fin >> population_count >> left_bound >> right_bound >> a >> b >> c >> precision;
    fin >> crossover_probability >> gene_mutation_probability >> steps;

    crossover_probability /= 100.0;
    gene_mutation_probability /= 100.0;

    find_closest_power_of_2(left_bound, right_bound);

    vector<bitset<M>> population;
    fout << "\nPopulatia initiala:\n";
    for (int i = 1; i <= population_count; ++i) {
        double actual_value = generate_random(left_bound, right_bound);
        population.push_back(encode(left_bound, right_bound, actual_value));
        fout << i << ": ";
        print_encoding(population.back());
        fout << ", x = " << setprecision(precision) << actual_value;
        fout << ", f = " << setprecision(precision) << get_fitness(actual_value) << '\n';
    }

    bool verbose = true;

    for (int cnt = 1; cnt <= steps; ++cnt) {
        population = selection(population, population_count, verbose);

        if (verbose) {
            fout << "\nDupa selectie (primul a fost selectat elitist):\n";
            for (int i = 0; i < (int) population.size(); ++i) {
                fout << i + 1 << ": ";
                print_encoding(population[i]);
                double curr_val = decode(left_bound, right_bound, population[i]);
                fout << ", x = " << setprecision(precision) << curr_val;
                fout << ", f = " << setprecision(P) << get_fitness(curr_val) << '\n';
            }
        }

        vector<bitset<M>> selected_population(population.begin() + 1, population.end());
        population.erase(population.begin() + 1, population.end());

        apply_crossover(selected_population, (int) selected_population.size(), crossover_probability, verbose);

        if (verbose) {
            fout << "\nDupa recombinare (primul a fost selectat elitist, nu a participat):\n";
            for (int i = 0; i < (int) selected_population.size(); ++i) {
                fout << i + 1 << ": ";
                print_encoding(selected_population[i]);
                double curr_val = decode(left_bound, right_bound, selected_population[i]);
                fout << ", x = " << setprecision(precision) << curr_val;
                fout << ", f = " << setprecision(P) << get_fitness(curr_val) << '\n';
            }
        }

        apply_mutations(selected_population, (int) selected_population.size(), gene_mutation_probability, verbose);

        if (verbose) {
            fout << "Dupa mutatie (primul a fost selectat elitist, nu a participat):\n";
            for (int i = 0; i < (int) selected_population.size(); ++i) {
                fout << i + 1 << ": ";
                print_encoding(selected_population[i]);
                double curr_val = decode(left_bound, right_bound, selected_population[i]);
                fout << ", x = " << setprecision(precision) << curr_val;
                fout << ", f = " << setprecision(P) << get_fitness(curr_val) << '\n';
            }
        }

        population.insert(population.end(), selected_population.begin(), selected_population.end());

        print_max(population, population_count, cnt);

        verbose = false;
    }
    return 0;
}
