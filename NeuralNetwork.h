#pragma once
#include <vector>
#include "Layer.h"

class NeuralNetwork {
public:
    std::vector<Layer> layers;  // Capas de la red neuronal

    NeuralNetwork(int inputSize, int hiddenLayers, int hiddenSize, int outputSize);

    void feedForward(const std::vector<double>& input);
    void backPropagation(const std::vector<double>& target);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs);
};
