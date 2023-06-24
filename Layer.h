#pragma once
#include <vector>
#include "Neuron.h"

class Layer {
public:
    std::vector<Neuron> neurons;  // Neuronas de la capa

    Layer(int size, int previousLayerSize);
    void setOutputs(const std::vector<double>& outputs);
    std::vector<double> getOutputs() const;
    void feedForward(const std::vector<double>& inputs);
    void calculateOutputLayerGradients(const std::vector<double>& target);
    void calculateHiddenLayerGradients(const std::vector<double>& nextLayerGamma, const std::vector<std::vector<double>>& nextLayerWeights);
    void updateWeights(const std::vector<double>& inputs);
    std::vector<double> getGamma() const;
    std::vector<std::vector<double>> getWeights() const;
    double calculateError(const std::vector<double>& target) const;
};
