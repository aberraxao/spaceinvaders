package nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class SpaceInvadersGeneticAlgorithm  {
    private static final int POPULATION_SIZE = 100;
    private static final int MAX_GENERATIONS = 100;
    private static final double MUTATION_RATE = 0.1;
    private static final double CROSSOVER_RATE = 0.8;
    private static final int TOURNAMENT_SIZE = 5;
    private static Random random = new Random();

    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private int seed;
    private AiController[] population;

    public SpaceInvadersGeneticAlgorithm(int inputDim, int hiddenDim, int outputDim, int seed) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.population = new AiController[POPULATION_SIZE];
        this.seed = seed;
    }

    private void initializePopulation(){
        for (int i = 0; i < POPULATION_SIZE; i++) {
            AiController network = new AiController(inputDim, hiddenDim, outputDim, seed);
            network.initializeWeights();
            network.calculateFitness();
            population[i] = network;
        }
    }

    public AiController train() {

        initializePopulation();

        // Evolve the population for a fixed number of generations
        for (int generation = 0; generation < MAX_GENERATIONS; generation++) {

            // Sort the population by fitness
            Arrays.sort(population);

            // Create the next generation
            AiController[] newPopulation = new AiController[POPULATION_SIZE];
            for (int pop = 0; pop < POPULATION_SIZE; pop++) {
                // Select two parents from the population
                AiController parent1 = selectParent(population);
                AiController parent2 = selectParent(population);
                // Crossover the parents to create a new child
                AiController child = crossover(parent1, parent2);
                // Mutate the child
                mutate(child);
                // Calculates the fitness
                child.calculateFitness();
                // Add the child to the new population
                newPopulation[pop] = child;
            }
            // Replace the old population with the new population
            population = newPopulation;

            // Print the best solution of this generation
            System.out.println("Generation " + generation + ": best solution -> " + population[0] + " with fitness " + population[0].getFitness());
        }
        // Print the best solution we found
        Arrays.sort(population);
        System.out.println("Best solution found: " + population[0] + " with fitness " + population[0].getFitness());

        return population[0];
    }

    private AiController selectParent(AiController[] population) {
        ArrayList<AiController> tournament = new ArrayList<>();
        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            tournament.add(population[random.nextInt(POPULATION_SIZE)]);
        }
        Collections.sort(tournament);
        return tournament.get(0);
    }

    private AiController crossover(AiController parent1, AiController parent2) {
        // Crossover two parents to create a new child
        AiController child = new AiController(inputDim, hiddenDim, outputDim, seed);

        // Crossover the input weights
        for (int i = 0; i < child.getInputDim(); i++) {
            for (int j = 0; j < child.getHiddenDim(); j++) {
                if (random.nextDouble() < CROSSOVER_RATE) {
                    child.getInputWeights()[i][j] = parent1.getInputWeights()[i][j];
                } else {
                    child.getInputWeights()[i][j] = parent2.getInputWeights()[i][j];
                }
            }
        }

        // Crossover the hidden biases and the output weights
        for (int i = 0; i < child.getHiddenDim(); i++) {
            if (random.nextDouble() < CROSSOVER_RATE) {
                child.getHiddenBiases()[i] = parent1.getHiddenBiases()[i];
            } else {
                child.getHiddenBiases()[i] = parent2.getHiddenBiases()[i];
            }
            for (int j = 0; j < child.getOutputDim(); j++) {
                if (random.nextDouble() < CROSSOVER_RATE) {
                    child.getOutputWeights()[i][j] = parent1.getOutputWeights()[i][j];
                } else {
                    child.getOutputWeights()[i][j] = parent2.getOutputWeights()[i][j];
                }
            }
        }

        // Crossover the output biases
        for (int i = 0; i < child.getOutputDim(); i++) {
            if (random.nextDouble() < CROSSOVER_RATE) {
                child.getOutputBiases()[i]= parent1.getOutputBiases()[i];
            } else {
                child.getOutputBiases()[i]= parent2.getOutputBiases()[i];
            }
        }

        return child;
    }

    private void mutate(AiController individual) {

        // Mutate the input weights
        for (int i = 0; i < individual.getInputDim(); i++) {
            for (int j = 0; j < individual.getHiddenDim(); j++) {
                if (random.nextDouble() < MUTATION_RATE) {
                    individual.getInputWeights()[i][j] += random.nextGaussian() * 0.1;
                }
            }
        }

        // Mutate the hidden biases and output weights
        for (int i = 0; i < individual.getHiddenDim(); i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                individual.getHiddenBiases()[i] += random.nextGaussian() * 0.1;
            }
            for (int j = 0; j < individual.getOutputDim(); j++) {
                if (random.nextDouble() < MUTATION_RATE) {
                    individual.getOutputWeights()[i][j] += random.nextGaussian() * 0.1;
                }
            }
        }

        // Mutate the output biases
        for (int i = 0; i < individual.getOutputDim(); i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                individual.getOutputBiases()[i] += random.nextGaussian() * 0.1;
            }
        }
    }

}

