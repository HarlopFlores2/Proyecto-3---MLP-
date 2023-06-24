#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <iostream>
using namespace std;

class Neuron
{
public:
    Neuron(int previousLayerSize);
    double activate(double input) const;
    double activateDerivative(double input) const;
    void setOutput(double value);
    double getOutput() const;
    void feedForward(const std::vector<double>& inputs);
    void calculateOutputGradient(double target);
    void calculateHiddenGradient(const std::vector<double>& nextLayerGamma, const std::vector<double>& nextLayerWeights);
    void updateWeights(const std::vector<double>& inputs);
    std::vector<double> getWeights() const;
    double calculateSquaredError(double target) const;
    double gradient;

private:
    std::vector<double> weights;
    double bias;
    double output;

    double sigmoid(double input) const;
    double sigmoidDerivative(double input) const;
    double relu(double input) const;
    double reluDerivative(double input) const;
};

#endif  // NEURON_H
