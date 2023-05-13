package controllers;

public class AiController implements GameController {

	private nn.AiController neuralNetwork;

	public AiController(nn.AiController neuralNetwork) {
		super();
		this.neuralNetwork = neuralNetwork;
	}

	@Override
	public double[] nextMove(double[] state) {
		return neuralNetwork.forward(state);
	}
}
