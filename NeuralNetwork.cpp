#include <vector>
#include "NeuralNetwork.h"
#include "Layer.h"

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenLayers, int hiddenSize, int outputSize) {
    // Crear la capa de entrada
    layers.push_back(Layer(inputSize, 0));

    // Crear las capas ocultas
    for (int i = 0; i < hiddenLayers; i++) {
        layers.push_back(Layer(hiddenSize, layers.back().neurons.size()));
    }

    // Crear la capa de salida
    layers.push_back(Layer(outputSize, layers.back().neurons.size()));
}




void NeuralNetwork::feedForward(const std::vector<double>& input) {
    layers[0].setOutputs(input);
    for (size_t i = 1; i < layers.size(); i++) {
        layers[i].feedForward(layers[i - 1].getOutputs());
    }
}






void NeuralNetwork::backPropagation(const std::vector<double>& target) {
    layers.back().calculateOutputLayerGradients(target);

    for (int i = layers.size() - 2; i > 0; i--) {
        layers[i].calculateHiddenLayerGradients(layers[i + 1].getGamma(), layers[i + 1].getWeights());
    }
    for (size_t i = 1; i < layers.size(); i++) {
        layers[i].updateWeights(layers[i - 1].getOutputs());
    }
}




void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;

        for (size_t i = 0; i < inputs.size(); i++) {
            feedForward(inputs[i]);
            backPropagation(targets[i]);
            totalError += layers.back().calculateError(targets[i]);
        }
    }
}


