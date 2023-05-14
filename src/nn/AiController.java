package nn;

import controllers.GameController;
import space.Board;

import java.util.Random;

public class AiController implements GameController, Comparable<AiController> {
    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private double[][] inputWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    private int seed;
    private double fitness;

    public AiController(int inputDim, int hiddenDim, int outputDim, int seed) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.inputWeights = new double[this.inputDim][this.hiddenDim];
        this.hiddenBiases = new double[this.hiddenDim];
        this.outputWeights = new double[this.hiddenDim][this.outputDim];
        this.outputBiases = new double[this.outputDim];

        this.seed = seed;

        calculateFitness();
    }

    public AiController(int inputDim, int hiddenDim, int outputDim, double[] values, int seed) {
        this(inputDim, hiddenDim, outputDim, seed);
        int offset;
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

        calculateFitness();
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

    public double getFitness(){
        return fitness;
    }

    public void initializeWeights() {
        // Randomly initialize weights and biases
        Random random = new Random();
        for (int i = 0; i < this.getInputDim(); i++) {
            for (int j = 0; j < this.getHiddenDim(); j++) {
                inputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < this.getHiddenDim(); i++) {
            hiddenBiases[i] = random.nextDouble() - 0.5;
            for (int j = 0; j < this.getOutputDim(); j++) {
                outputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < this.getOutputDim(); i++) {
            outputBiases[i] = random.nextDouble() - 0.5;
        }
    }

    private void calculateFitness() {
        Board b = new Board(this);
        b.setSeed(seed);
        b.run();
        fitness = b.getFitness();
/*
        private double evaluateFitness(AiController network, Board board) {
            // Simulate the game using the network and board
            Board simulator = new Board(network);
            simulator.setSeed(5);
            simulator.run();

            // Calculate fitness based on game score or other metrics
            double score = simulator.getFitness();
            // Other fitness calculations can be done here

            return score;
        }
*/
    }

    public double[] forward(double[] input) {
        // TODO: check with Diogo
        // Compute output given input
        double[] hidden = new double[this.getHiddenDim()];
        for (int i = 0; i < this.getHiddenDim(); i++) {
            double sum = 0.0;
            for (int j = 0; j < this.getInputDim(); j++) {
                double d = input[j];
                sum += d * inputWeights[j][i];
            }
            hidden[i] = Math.max(0.0, sum + hiddenBiases[i]);
        }
        double[] output = new double[this.getOutputDim()];
        for (int i = 0; i < this.getOutputDim(); i++) {
            double sum = 0.0;
            for (int j = 0; j < this.getHiddenDim(); j++) {
                sum += hidden[j] * outputWeights[j][i];
            }
            output[i] = Math.exp(sum + outputBiases[i]);
        }
        double sum = 0.0;
        for (int i = 0; i < this.getOutputDim(); i++) {
            sum += output[i];
        }
        for (int i = 0; i < this.getOutputDim(); i++) {
            output[i] /= sum;
        }
        return output;
    }

    @Override
    public double[] nextMove(double[] currentState) {
        return forward(currentState);
    }

    @Override
    public int compareTo(AiController aiController) {
        return Double.compare(aiController.getFitness(), this.getFitness());
    }
}