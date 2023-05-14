package main;

import nn.SpaceInvadersGeneticAlgorithm;
import space.Commons;
import space.SpaceInvaders;

public class PlayAiController {
	public static void main(String[] args) {
		int inputDim = Commons.STATE_SIZE;
		int hiddenDim = 25;
		int outputDim = Commons.NUM_ACTIONS;
		int seed = 5;

		SpaceInvadersGeneticAlgorithm geneticAlgorithm = new SpaceInvadersGeneticAlgorithm(inputDim, hiddenDim, outputDim, seed);

		SpaceInvaders.showControllerPlaying(geneticAlgorithm.train(),seed);
	}
}
