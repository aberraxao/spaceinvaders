package nn;

import space.Board;

import java.util.Random;

public class SpaceInvadersGeneticAlgorithm {
    private static final int POPULATION_SIZE = 100;
    private static final double MUTATION_RATE = 0.1;
    private static final double CROSSOVER_RATE = 0.8;
    private static final int TOURNAMENT_SIZE = 5;
    private static final int MAX_GENERATIONS = 100;

    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private AiController[] population;

    public SpaceInvadersGeneticAlgorithm(int inputDim, int hiddenDim, int outputDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.population = new AiController[POPULATION_SIZE];
    }

    public void initializePopulation() {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            AiController network = new AiController(inputDim, hiddenDim, outputDim);
            network.initializeWeights();
            population[i] = network;
        }
    }

    public AiController evolve(Board board) {
        initializePopulation();

        for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
            System.out.println("Generation: " + generation);

            AiController[] newPopulation = new AiController[POPULATION_SIZE];

            for (int i = 0; i < POPULATION_SIZE; i++) {
                AiController parent1 = selectParent();
                AiController parent2 = selectParent();

                AiController offspring = crossover(parent1, parent2);

                mutate(offspring);

                // Evaluate the network's performance on the game board
                double fitness = evaluateFitness(offspring, board);
                offspring.setFitness(fitness);

                newPopulation[i] = offspring;
            }

            population = newPopulation;
        }

        return getBestNetwork();
    }

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

    private AiController selectParent() {
        Random random = new Random();
        AiController bestNetwork = population[random.nextInt(POPULATION_SIZE)];

        for (int i = 0; i < TOURNAMENT_SIZE; i++) {
            AiController network = population[random.nextInt(POPULATION_SIZE)];
            if (network.getFitness() > bestNetwork.getFitness()) {
                bestNetwork = network;
            }
        }

        return bestNetwork;
    }

    private AiController crossover(AiController parent1, AiController parent2) {
        AiController offspring = new AiController(inputDim, hiddenDim, outputDim);

        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                if (Math.random() <= CROSSOVER_RATE) {
                    offspring.getInputWeights()[i][j] = parent1.getInputWeights()[i][j];
                } else {
                    offspring.getInputWeights()[i][j] = parent2.getInputWeights()[i][j];
                }
            }
        }

        for (int i = 0; i < hiddenDim; i++) {
            if (Math.random() <= CROSSOVER_RATE) {
                offspring.getHiddenBiases()[i] = parent1.getHiddenBiases()[i];
            } else {
                offspring.getHiddenBiases()[i] = parent2.getHiddenBiases()[i];
            }
            for (int j = 0; j < outputDim; j++) {
                if (Math.random() <= CROSSOVER_RATE) {
                    offspring.getOutputWeights()[i][j] = parent1.getOutputWeights()[i][j];
                } else {
                    offspring.getOutputWeights()[i][j] = parent2.getOutputWeights()[i][j];
                }
            }
        }

        for (int i = 0; i < outputDim; i++) {
            if (Math.random() <= CROSSOVER_RATE) {
                offspring.getOutputBiases()[i] = parent1.getOutputBiases()[i];
            } else {
                offspring.getOutputBiases()[i] = parent2.getOutputBiases()[i];
            }
        }

        return offspring;
    }

    private void mutate(AiController network) {
        Random random = new Random();

        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                if (Math.random() <= MUTATION_RATE) {
                    network.getInputWeights()[i][j] += random.nextGaussian() * 0.1;
                }
            }
        }

        for (int i = 0; i < hiddenDim; i++) {
            if (Math.random() <= MUTATION_RATE) {
                network.getHiddenBiases()[i] += random.nextGaussian() * 0.1;
            }
            for (int j = 0; j < outputDim; j++) {
                if (Math.random() <= MUTATION_RATE) {
                    network.getOutputWeights()[i][j] += random.nextGaussian() * 0.1;
                }
            }
        }

        for (int i = 0; i < outputDim; i++) {
            if (Math.random() <= MUTATION_RATE) {
                network.getOutputBiases()[i] += random.nextGaussian() * 0.1;
            }
        }
    }


    private AiController getBestNetwork() {
        AiController bestNetwork = population[0];
        double bestFitness = bestNetwork.getFitness();

        for (int i = 1; i < POPULATION_SIZE; i++) {
            double fitness = population[i].getFitness();
            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestNetwork = population[i];
            }
        }

        return bestNetwork;
    }
}

