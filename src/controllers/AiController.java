package controllers;

import nn.SpaceInvadersNeuralNetwork;

public class AiController implements GameController {

	private SpaceInvadersNeuralNetwork neuralNetwork;

	public AiController(SpaceInvadersNeuralNetwork neuralNetwork) {
		super();
		this.neuralNetwork = neuralNetwork;
	}

	@Override
	public double[] nextMove(double[] state) {
		return neuralNetwork.predict(state);
	}
}
