#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <stdexcept>

using namespace std;

// Constants
const int MAX_BITS = 100;
const int OUTPUT_PRECISION = 20;
const long long INF = 1e18;

// Configuration parameters
struct GeneticAlgorithmConfig {
    int populationSize;
    int leftBound;
    int rightBound;
    int a, b, c; // Quadratic function coefficients
    int precision;
    double crossoverProbability;
    double mutationProbability;
    int generations;
};

// Chromosome representation and utilities
class Chromosome {
private:
    bitset<MAX_BITS> encoding;
    int encodingLength;
    
public:
    Chromosome() = default;
    
    Chromosome(const bitset<MAX_BITS>& enc, int len) 
        : encoding(enc), encodingLength(len) {}
        
    void setEncoding(const bitset<MAX_BITS>& enc) { encoding = enc; }
    bitset<MAX_BITS> getEncoding() const { return encoding; }
    
    void setEncodingLength(int len) { encodingLength = len; }
    int getEncodingLength() const { return encodingLength; }
    
    string getEncodingString() const {
        return encoding.to_string().substr(MAX_BITS - encodingLength);
    }
};

// Genetic Algorithm class
class GeneticAlgorithm {
private:
    GeneticAlgorithmConfig config;
    long long closestPowerOf2;
    int encodingLength;
    mt19937 randomGenerator;
    
    // Initialize random number generator
    void initRandomGenerator() {
        random_device rd;
        randomGenerator = mt19937(rd());
    }
    
    // Find the closest power of 2 for the given range and precision
    void calculateClosestPowerOf2() {
        long long pw10 = 1;
        for (int i = 1; i <= config.precision; ++i) {
            pw10 *= 10;
        }

        long long intervalsCount = 1LL * (config.rightBound - config.leftBound) * pw10;
        closestPowerOf2 = 1;
        encodingLength = 0;
        
        while (closestPowerOf2 < intervalsCount) {
            closestPowerOf2 <<= 1LL;
            encodingLength++;
        }
    }
    
    // Encode a real value to binary representation
    bitset<MAX_BITS> encodeValue(double value) const {
        long long intervalIdx = floor((value - config.leftBound) * closestPowerOf2 / 
                                    (config.rightBound - config.leftBound));
        return bitset<MAX_BITS>(intervalIdx);
    }
    
    // Decode binary representation to real value
    double decodeValue(const bitset<MAX_BITS>& encoding) const {
        long long intervalIdx = encoding.to_ullong();
        double step = (config.rightBound - config.leftBound) / static_cast<double>(closestPowerOf2);
        return config.leftBound + intervalIdx * step;
    }
    
    // Generate random double in [start, end]
    double generateRandomDouble(double start, double end) {
        uniform_real_distribution<> dis(start, end);
        return dis(randomGenerator);
    }
    
    // Calculate fitness using the quadratic function
    double calculateFitness(double value) const {
        return config.a * value * value + config.b * value + config.c;
    }
    
    // Tournament selection
    vector<Chromosome> tournamentSelection(const vector<Chromosome>& population, bool verbose) {
        vector<Chromosome> selectedPopulation;
        
        // Find best chromosome (elitism)
        auto bestIt = max_element(population.begin(), population.end(),
            [this](const Chromosome& a, const Chromosome& b) {
                return calculateFitness(decodeValue(a.getEncoding())) < 
                       calculateFitness(decodeValue(b.getEncoding()));
            });
        
        selectedPopulation.push_back(*bestIt);
        
        if (verbose) {
            int pos = distance(population.begin(), bestIt);
            cout << "Chromosome " << pos + 1 << " was selected through elitist selection\n";
        }
        
        // Calculate selection probabilities
        vector<pair<double, int>> fitnessValues;
        double minFitness = 0.0;
        
        for (size_t i = 0; i < population.size(); ++i) {
            if (i == static_cast<size_t>(distance(population.begin(), bestIt))) continue;
            
            double fitness = calculateFitness(decodeValue(population[i].getEncoding()));
            fitnessValues.emplace_back(fitness, i);
            minFitness = min(minFitness, fitness);
        }
        
        // Adjust fitness values to be positive
        double sumFitness = 0.0;
        for (auto& [fitness, idx] : fitnessValues) {
            fitness += abs(minFitness) + 1e-6; // Small epsilon to avoid zero
            sumFitness += fitness;
        }
        
        // Create roulette wheel intervals
        vector<double> intervals = {0.0};
        for (const auto& [fitness, idx] : fitnessValues) {
            intervals.push_back(intervals.back() + fitness / sumFitness);
        }
        
        // Select chromosomes using roulette wheel
        for (size_t i = 0; i < population.size() - 1; ++i) {
            double randVal = generateRandomDouble(0.0, 1.0);
            auto it = upper_bound(intervals.begin(), intervals.end(), randVal);
            size_t selectedIdx = distance(intervals.begin(), it) - 1;
            
            selectedPopulation.push_back(population[fitnessValues[selectedIdx].second]);
        }
        
        return selectedPopulation;
    }
    
    // Perform crossover between chromosomes
    void performCrossover(vector<Chromosome>& population, bool verbose) {
        vector<size_t> selectedIndices;
        
        // Select chromosomes for crossover based on probability
        for (size_t i = 0; i < population.size(); ++i) {
            double randVal = generateRandomDouble(0.0, 1.0);
            if (randVal <= config.crossoverProbability) {
                selectedIndices.push_back(i);
            }
        }
        
        // Perform crossover on selected chromosomes
        if (selectedIndices.size() >= 2) {
            shuffle(selectedIndices.begin(), selectedIndices.end(), randomGenerator);
            
            for (size_t i = 0; i + 1 < selectedIndices.size(); i += 2) {
                size_t idx1 = selectedIndices[i];
                size_t idx2 = selectedIndices[i + 1];
                
                int crossoverPoint = generateRandomDouble(0, encodingLength - 1);
                
                if (verbose) {
                    cout << "Crossover between chromosomes " << idx1 + 1 << " and " << idx2 + 1 
                         << " at point " << crossoverPoint << ":\n";
                    cout << "Before: " << population[idx1].getEncodingString() << " | " 
                         << population[idx2].getEncodingString() << "\n";
                }
                
                // Perform crossover
                for (int j = 0; j < encodingLength - crossoverPoint; ++j) {
                    bool temp = population[idx1].getEncoding()[j];
                    population[idx1].getEncoding()[j] = population[idx2].getEncoding()[j];
                    population[idx2].getEncoding()[j] = temp;
                }
                
                if (verbose) {
                    cout << "After:  " << population[idx1].getEncodingString() << " | " 
                         << population[idx2].getEncodingString() << "\n";
                }
            }
        }
    }
    
    // Apply mutations to the population
    void applyMutations(vector<Chromosome>& population, bool verbose) {
        vector<size_t> mutatedIndices;
        
        for (size_t i = 0; i < population.size(); ++i) {
            bool mutated = false;
            
            for (int j = 0; j < encodingLength; ++j) {
                double randVal = generateRandomDouble(0.0, 1.0);
                if (randVal <= config.mutationProbability) {
                    population[i].getEncoding().flip(j);
                    mutated = true;
                }
            }
            
            if (mutated && verbose) {
                mutatedIndices.push_back(i);
            }
        }
        
        if (verbose && !mutatedIndices.empty()) {
            cout << "Mutated chromosomes: ";
            for (size_t idx : mutatedIndices) {
                cout << idx + 1 << " ";
            }
            cout << "\n";
        }
    }
    
public:
    GeneticAlgorithm(const GeneticAlgorithmConfig& cfg) : config(cfg) {
        initRandomGenerator();
        calculateClosestPowerOf2();
    }
    
    // Run the genetic algorithm
    void run(ostream& outputStream) {
        // Initialize population
        vector<Chromosome> population(config.populationSize);
        
        outputStream << "\nInitial population:\n";
        for (int i = 0; i < config.populationSize; ++i) {
            double value = generateRandomDouble(config.leftBound, config.rightBound);
            population[i].setEncoding(encodeValue(value));
            population[i].setEncodingLength(encodingLength);
            
            outputStream << i + 1 << ": " << population[i].getEncodingString()
                         << ", x = " << fixed << setprecision(config.precision) << value
                         << ", f = " << calculateFitness(value) << "\n";
        }
        
        bool verbose = true;
        
        for (int generation = 1; generation <= config.generations; ++generation) {
            // Selection
            vector<Chromosome> selectedPopulation = tournamentSelection(population, verbose);
            
            if (verbose) {
                outputStream << "\nAfter selection (first was elite):\n";
                for (size_t i = 0; i < selectedPopulation.size(); ++i) {
                    double value = decodeValue(selectedPopulation[i].getEncoding());
                    outputStream << i + 1 << ": " << selectedPopulation[i].getEncodingString()
                                << ", x = " << fixed << setprecision(config.precision) << value
                                << ", f = " << calculateFitness(value) << "\n";
                }
            }
            
            // Crossover
            vector<Chromosome> crossoverPopulation(selectedPopulation.begin() + 1, 
                                                 selectedPopulation.end());
            performCrossover(crossoverPopulation, verbose);
            
            if (verbose) {
                outputStream << "\nAfter crossover:\n";
                for (size_t i = 0; i < crossoverPopulation.size(); ++i) {
                    double value = decodeValue(crossoverPopulation[i].getEncoding());
                    outputStream << i + 1 << ": " << crossoverPopulation[i].getEncodingString()
                                << ", x = " << fixed << setprecision(config.precision) << value
                                << ", f = " << calculateFitness(value) << "\n";
                }
            }
            
            // Mutation
            applyMutations(crossoverPopulation, verbose);
            
            if (verbose) {
                outputStream << "\nAfter mutation:\n";
                for (size_t i = 0; i < crossoverPopulation.size(); ++i) {
                    double value = decodeValue(crossoverPopulation[i].getEncoding());
                    outputStream << i + 1 << ": " << crossoverPopulation[i].getEncodingString()
                                << ", x = " << fixed << setprecision(config.precision) << value
                                << ", f = " << calculateFitness(value) << "\n";
                }
            }
            
            // Combine elite with new population
            population.clear();
            population.push_back(selectedPopulation[0]); // Keep elite
            population.insert(population.end(), crossoverPopulation.begin(), crossoverPopulation.end());
            
            // Calculate statistics
            double maxFitness = -INF;
            double sumFitness = 0.0;
            
            for (const auto& chromosome : population) {
                double fitness = calculateFitness(decodeValue(chromosome.getEncoding()));
                maxFitness = max(maxFitness, fitness);
                sumFitness += fitness;
            }
            
            outputStream << "Generation " << generation << ":\n";
            outputStream << "Max fitness: " << fixed << setprecision(OUTPUT_PRECISION) << maxFitness << "\n";
            outputStream << "Mean fitness: " << fixed << setprecision(OUTPUT_PRECISION) 
                         << sumFitness / population.size() << "\n\n";
            
            verbose = false; // Only show details for first generation
        }
    }
};

int main() {
    try {
        ifstream inputFile("input.in");
        ofstream outputFile("output.out");
        
        if (!inputFile.is_open() || !outputFile.is_open()) {
            throw runtime_error("Failed to open input or output file");
        }
        
        GeneticAlgorithmConfig config;
        inputFile >> config.populationSize >> config.leftBound >> config.rightBound
                 >> config.a >> config.b >> config.c >> config.precision
                 >> config.crossoverProbability >> config.mutationProbability
                 >> config.generations;
        
        // Convert probabilities from percentages
        config.crossoverProbability /= 100.0;
        config.mutationProbability /= 100.0;
        
        GeneticAlgorithm ga(config);
        ga.run(outputFile);
        
        return 0;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}