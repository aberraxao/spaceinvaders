package nn;

import java.util.Random;

public class SpaceInvadersNeuralNetwork {
    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private double[][] inputWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    public SpaceInvadersNeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.inputWeights = new double[inputDim][hiddenDim];
        this.hiddenBiases = new double[hiddenDim];
        this.outputWeights = new double[hiddenDim][outputDim];
        this.outputBiases = new double[outputDim];
    }

    public SpaceInvadersNeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
        this(inputDim, hiddenDim, outputDim);
        int offset = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                inputWeights[i][j] = values[i * hiddenDim + j];
            }
        }
        offset = inputDim * hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = values[offset + i];
        }
        offset += hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = values[offset + i * outputDim + j];
            }
        }
        offset += hiddenDim * outputDim;
        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = values[offset + i];
        }
    }

    public int getChromosomeSize() {
        return inputWeights.length * inputWeights[0].length + hiddenBiases.length
                + outputWeights.length * outputWeights[0].length + outputBiases.length;
    }

    public double[] getChromosome() {
        double[] chromosome = new double[getChromosomeSize()];
        int offset = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                chromosome[i * hiddenDim + j] = inputWeights[i][j];
            }
        }
        offset = inputDim * hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            chromosome[offset + i] = hiddenBiases[i];
        }
        offset += hiddenDim;
        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                chromosome[offset + i * outputDim + j] = outputWeights[i][j];
            }
        }
        offset += hiddenDim * outputDim;
        for (int i = 0; i < outputDim; i++) {
            chromosome[offset + i] = outputBiases[i];
        }

        return chromosome;
    }

    public void initializeWeights() {
        // Randomly initialize weights and biases
        Random random = new Random();
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                inputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < hiddenDim; i++) {
            hiddenBiases[i] = random.nextDouble() - 0.5;
            for (int j = 0; j < outputDim; j++) {
                outputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = random.nextDouble() - 0.5;
        }
    }

    public int getInputDim() {
        return inputDim;
    }

    public int getHiddenDim() {
        return hiddenDim;
    }

    public int getOutputDim() {
        return outputDim;
    }

    public double[][] getInputWeights() {
        return inputWeights;
    }

    public double[] getHiddenBiases() {
        return hiddenBiases;
    }

    public double[][] getOutputWeights() {
        return outputWeights;
    }

    public double[] getOutputBiases() {
        return outputBiases;
    }

    public double[] forward(double[] input) {
        // Compute output given input
        double[] hidden = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputDim; j++) {
                double d = input[j];
                sum += d * inputWeights[j][i];
            }
            hidden[i] = Math.max(0.0, sum + hiddenBiases[i]);
        }
        double[] output = new double[outputDim];
        for (int i = 0; i < outputDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenDim; j++) {
                sum += hidden[j] * outputWeights[j][i];
            }
            output[i] = Math.exp(sum + outputBiases[i]);
        }
        double sum = 0.0;
        for (int i = 0; i < outputDim; i++) {
            sum += output[i];
        }
        for (int i = 0; i < outputDim; i++) {
            output[i] /= sum;
        }
        return output;
    }

    public double[] predict(double[] input) {
        return forward(input);
    }

    public double getFitness() {
        return 1.0;
    }
}