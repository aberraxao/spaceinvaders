package nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.logging.Level;

import static main.PlayAiController.logger;

public class AiGeneticEvolution {
    private static final int POPULATION_SIZE = 100;
    private static final int MAX_GENERATIONS = 500;
    private static final double MUTATION_RATE = 0.1;
    private static final double CROSSOVER_RATE = 0.8;
    private static final int TOURNAMENT_SIZE = 2;
    private Random random;
    private int seed;
    private AiFeedForwardController[] population = new AiFeedForwardController[POPULATION_SIZE];
    private AiFeedForwardController bestNetWork;

    public AiGeneticEvolution(int seed) {
        this.seed = seed;
        this.random = new Random(seed);
        initializePopulation();
    }

    private void initializePopulation() {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            AiFeedForwardController network = new AiFeedForwardController(seed);
            network.initializeWeightsAndBiases();
            network.calculateAndSetFitness();
            population[i] = network;
        }
        Arrays.sort(population);
        bestNetWork = population[0];
        logger.log(Level.INFO, "First best network found {0} with fitness {1}", new Object[]{bestNetWork, bestNetWork.getFitness()});
    }
    private void updateBestNetwork(AiFeedForwardController currentNetwork) {
        if (currentNetwork.compareTo(bestNetWork) < 0) {
            bestNetWork = currentNetwork;
            logger.log(Level.INFO, "New best network found {0} with fitness {1}", new Object[]{bestNetWork, bestNetWork.getFitness()});
        }
    }

    public AiFeedForwardController train() {

        for (int generation = 0; generation < MAX_GENERATIONS; generation++) {

            logger.log(Level.INFO, "------- Generation {0} -------", generation);

            AiFeedForwardController[] newPopulation = new AiFeedForwardController[POPULATION_SIZE];
            for (int pop = 0; pop < POPULATION_SIZE; pop++) {
                AiFeedForwardController parent1 = selectParent(population);
                AiFeedForwardController parent2 = selectParent(population);
                AiFeedForwardController child = crossover(parent1, parent2);
                mutate(child);
                child.calculateAndSetFitness();
                newPopulation[pop] = child;
                updateBestNetwork(child);
            }
            population = newPopulation;
        }

        return bestNetWork;
    }

    private AiFeedForwardController selectParent(AiFeedForwardController[] population) {
        ArrayList<AiFeedForwardController> tournament = new ArrayList<>();
        for (int i = 0; i < TOURNAMENT_SIZE; i++)
            tournament.add(population[random.nextInt(POPULATION_SIZE)]);

        Collections.sort(tournament);
        return tournament.get(0);
    }

    private AiFeedForwardController crossover(AiFeedForwardController parent1, AiFeedForwardController parent2) {

        AiFeedForwardController child = new AiFeedForwardController(seed);

        crossoverInputWeights(child, parent1, parent2);
        crossoverHiddenBiasesAndOutputWeights(child, parent1, parent2);
        crossoverOutputBiases(child, parent1, parent2);

        return child;
    }

    private void crossoverInputWeights(AiFeedForwardController child, AiFeedForwardController parent1, AiFeedForwardController parent2) {
        for (int i = 0; i < child.getInputDim(); i++) {
            for (int j = 0; j < child.getHiddenDim(); j++) {
                if (random.nextDouble() < CROSSOVER_RATE) child.setInputWeights(i, j, parent1.getInputWeights(i, j));
                else child.setInputWeights(i, j, parent2.getInputWeights(i, j));
            }
        }
    }

    private void crossoverHiddenBiasesAndOutputWeights(AiFeedForwardController child, AiFeedForwardController parent1, AiFeedForwardController parent2) {
        for (int i = 0; i < child.getHiddenDim(); i++) {
            if (random.nextDouble() < CROSSOVER_RATE) child.setHiddenBiases(i, parent1.getHiddenBiases(i));
            else child.setHiddenBiases(i, parent2.getHiddenBiases(i));
            for (int j = 0; j < child.getOutputDim(); j++) {
                if (random.nextDouble() < CROSSOVER_RATE)
                    child.setOutputWeights(i, j, parent1.getOutputWeights(i, j));
                else child.setOutputWeights(i, j, parent2.getOutputWeights(i, j));
            }
        }
    }

    private void crossoverOutputBiases(AiFeedForwardController child, AiFeedForwardController parent1, AiFeedForwardController parent2) {
        for (int i = 0; i < child.getOutputDim(); i++) {
            if (random.nextDouble() < CROSSOVER_RATE) child.setOutputBiases(i, parent1.getOutputBiases(i));
            else child.setOutputBiases(i, parent2.getOutputBiases(i));
        }
    }

    private void mutate(AiFeedForwardController child) {
        mutateInputWeights(child);
        mutateHiddenBiasesAndOutputWeights(child);
        mutateOutputBiases(child);
    }

    private void mutateInputWeights(AiFeedForwardController child) {
        for (int i = 0; i < child.getInputDim(); i++)
            for (int j = 0; j < child.getHiddenDim(); j++)
                if (random.nextDouble() < MUTATION_RATE)
                    child.setInputWeights(i, j, child.getInputWeights(i, j) + random.nextGaussian() * 0.1);
    }

    private void mutateHiddenBiasesAndOutputWeights(AiFeedForwardController child) {
        for (int i = 0; i < child.getHiddenDim(); i++) {
            if (random.nextDouble() < MUTATION_RATE)
                child.setHiddenBiases(i, child.getHiddenBiases(i) + random.nextGaussian() * 0.1);
            for (int j = 0; j < child.getOutputDim(); j++)
                if (random.nextDouble() < MUTATION_RATE)
                    child.setOutputWeights(i, j, child.getOutputWeights(i, j) + random.nextGaussian() * 0.1);
        }
    }

    private void mutateOutputBiases(AiFeedForwardController child) {
        for (int i = 0; i < child.getOutputDim(); i++)
            if (random.nextDouble() < MUTATION_RATE)
                child.setOutputBiases(i, child.getOutputBiases(i) + random.nextGaussian() * 0.1);
    }
}

