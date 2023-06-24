#include <vector>
#include <cstdlib>
#include <cmath>
#include "Neuron.h"

Neuron::Neuron(int previousLayerSize)
{
    // Inicializar los pesos con valores aleatorios
    for (int i = 0; i < previousLayerSize; i++)
    {
        weights.push_back(((double)rand() / RAND_MAX) * 2 - 1);  
    }
    bias = ((double)rand() / RAND_MAX) * 2 - 1; 
}

double Neuron::activate(double input) const
{
    return relu(input);  
}

double Neuron::activateDerivative(double input) const
{
    return reluDerivative(input);  
}

void Neuron::setOutput(double value)
{
    output = value;
}

double Neuron::getOutput() const
{
    return output;
}

void Neuron::feedForward(const std::vector<double>& inputs)
{
    double sum = bias;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        sum += inputs[i] * weights[i];
    }
    output = activate(sum);
}

void Neuron::calculateOutputGradient(double target)
{
    double delta = target - output;
    double derivative = activateDerivative(output);
    gradient = delta * derivative;
}

void Neuron::calculateHiddenGradient(const std::vector<double>& nextLayerGamma, const std::vector<double>& nextLayerWeights)
{
    double sum = 0.0;
    for (size_t i = 0; i < nextLayerGamma.size(); i++)
    {
        sum += nextLayerGamma[i] * nextLayerWeights[i];
    }
    double derivative = activateDerivative(output);
    gradient = sum * derivative;
}

void Neuron::updateWeights(const std::vector<double>& inputs)
{
    for (size_t i = 0; i < weights.size(); i++)
    {
        weights[i] += inputs[i] * gradient;
    }
    bias += gradient;
}

std::vector<double> Neuron::getWeights() const
{
    return weights;
}

double Neuron::calculateSquaredError(double target) const
{
    double error = 0.5 * std::pow(target - output, 2);
    return error;
}

double Neuron::sigmoid(double input) const
{
    return 1 / (1 + std::exp(-input));
}

double Neuron::sigmoidDerivative(double input) const
{
    double sigmoid = 1 / (1 + std::exp(-input));
    return sigmoid * (1 - sigmoid);
}

double Neuron::relu(double input) const
{
    return std::max(0.0, input);
}

double Neuron::reluDerivative(double input) const
{
    return input > 0 ? 1.0 : 0.0;
}
