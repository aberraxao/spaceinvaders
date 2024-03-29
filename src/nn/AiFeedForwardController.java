package nn;

import controllers.GameController;
import space.Board;
import space.Commons;

import java.util.Random;
import java.util.logging.Level;

import static main.PlayAiController.logger;

public class AiFeedForwardController implements GameController, Comparable<AiFeedForwardController> {
    private static final int INPUT_DIM = Commons.STATE_SIZE;
    private static final int HIDDEN_DIM = 25;
    private static final int OUTPUT_DIM = Commons.NUM_ACTIONS;
    private double[][] inputWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;
    private double fitness;

    public AiFeedForwardController() {
        this.inputWeights = new double[this.getInputDim()][this.getHiddenDim()];
        this.hiddenBiases = new double[this.getHiddenDim()];
        this.outputWeights = new double[this.getHiddenDim()][this.getOutputDim()];
        this.outputBiases = new double[this.getOutputDim()];
    }

    public int getInputDim() {
        return INPUT_DIM;
    }

    public int getHiddenDim() {
        return HIDDEN_DIM;
    }

    public int getOutputDim() {
        return OUTPUT_DIM;
    }

    public void setInputWeights(int i, int j, double inputWeights) {
        this.inputWeights[i][j] = inputWeights;
    }

    public double getInputWeights(int i, int j) {
        return inputWeights[i][j];
    }

    public void setHiddenBiases(int i, double hiddenBiases) {
        this.hiddenBiases[i] = hiddenBiases;
    }

    public double getHiddenBiases(int i) {
        return hiddenBiases[i];
    }

    public void setOutputWeights(int i, int j, double outputWeights) {
        this.outputWeights[i][j] = outputWeights;
    }

    public double getOutputWeights(int i, int j) {
        return outputWeights[i][j];
    }

    public void setOutputBiases(int i, double outputBiases) {
        this.outputBiases[i] = outputBiases;
    }

    public double getOutputBiases(int i) {
        return outputBiases[i];
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public double getFitness() {
        return fitness;
    }

    public void initializeWeightsAndBiases() {
        Random random = new Random();
        logger.log(Level.INFO, "-> Feed Forward weights generated with random {0}", random);

        for (int i = 0; i < this.getInputDim(); i++)
            for (int j = 0; j < this.getHiddenDim(); j++)
                setInputWeights(i, j, random.nextDouble() - 0.5);

        for (int i = 0; i < this.getHiddenDim(); i++) {
            setHiddenBiases(i, random.nextDouble() - 0.5);
            for (int j = 0; j < this.getOutputDim(); j++)
                setOutputWeights(i, j, random.nextDouble() - 0.5);
        }
        for (int i = 0; i < this.getOutputDim(); i++)
            setOutputBiases(i, random.nextDouble() - 0.5);
    }

    public void calculateAndSetFitness(int seed) {
        Board b = new Board(this);
        b.setSeed(seed);
        b.run();
        this.setFitness(b.getFitness());
    }

    public double[] forward(double[] input) {
        double[] hidden = new double[this.getHiddenDim()];
        for (int j = 0; j < this.getHiddenDim(); j++) {
            double sum = 0.0;
            for (int i = 0; i < this.getInputDim(); i++) {
                double d = input[i];
                sum += d * getInputWeights(i, j);
            }
            hidden[j] = Math.max(0.0, sum + getHiddenBiases(j));
        }
        double[] output = new double[this.getOutputDim()];
        for (int j = 0; j < this.getOutputDim(); j++) {
            double sum = 0.0;
            for (int i = 0; i < this.getHiddenDim(); i++) {
                sum += hidden[i] * getOutputWeights(i, j);
            }
            output[j] = Math.exp(sum + getOutputBiases(j));
        }

        double sum = 0.0;
        for (int i = 0; i < this.getOutputDim(); i++) {
            sum += output[i];
        }

        if (sum == 0) throw new ArithmeticException("Division by zero!");
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
    public int compareTo(AiFeedForwardController aiFeedForwardController) {
        return Double.compare(aiFeedForwardController.getFitness(), this.getFitness());
    }
}