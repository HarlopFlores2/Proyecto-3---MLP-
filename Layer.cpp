#include <vector>
#include "Layer.h"
#include "Neuron.h"

Layer::Layer(int size, int previousLayerSize) {
    for (int i = 0; i < size; i++) {
        neurons.push_back(Neuron(previousLayerSize));
    }
}

void Layer::setOutputs(const std::vector<double>& outputs) {
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].setOutput(outputs[i]);
    }
}

std::vector<double> Layer::getOutputs() const {
    std::vector<double> outputs(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        outputs[i] = neurons[i].getOutput();
    }
    return outputs;
}

void Layer::feedForward(const std::vector<double>& inputs) {
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].feedForward(inputs);
    }
}

void Layer::calculateOutputLayerGradients(const std::vector<double>& target) {
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].calculateOutputGradient(target[i]);
    }
}

void Layer::calculateHiddenLayerGradients(const std::vector<double>& nextLayerGamma, const std::vector<std::vector<double>>& nextLayerWeights) {
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].calculateHiddenGradient(nextLayerGamma, nextLayerWeights[i]);
    }
}

void Layer::updateWeights(const std::vector<double>& inputs) {
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].updateWeights(inputs);
    }
}

std::vector<double> Layer::getGamma() const {
    std::vector<double> gamma(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        gamma[i] = neurons[i].gradient;
    }
    return gamma;
}


std::vector<std::vector<double>> Layer::getWeights() const {
    std::vector<std::vector<double>> weights(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        weights[i] = neurons[i].getWeights();
    }
    return weights;
}

double Layer::calculateError(const std::vector<double>& target) const {
    double error = 0.0;
    for (size_t i = 0; i < neurons.size(); i++) {
        error += neurons[i].calculateSquaredError(target[i]);
    }
    return error / neurons.size();
}

