package main;

import java.util.logging.Level;
import java.util.logging.Logger;

import nn.AiGeneticEvolution;
import space.SpaceInvaders;

public class PlayAiController {

    public static final Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    public static void main(String[] args) {

        int seed = 5;
        logger.log(Level.INFO, "Seed: {0}", seed);
        AiGeneticEvolution nn = new AiGeneticEvolution(seed);
        SpaceInvaders.showControllerPlaying(nn.train(), seed);
    }
}
