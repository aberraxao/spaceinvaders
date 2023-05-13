package space;

import java.awt.EventQueue;

import javax.swing.JFrame;

import controllers.AiController;
import controllers.GameController;
import nn.SpaceInvadersGeneticAlgorithm;
import nn.SpaceInvadersNeuralNetwork;

public class SpaceInvaders extends JFrame {

	private Board board;

	private SpaceInvadersNeuralNetwork neuralNetwork;

	public SpaceInvaders() {
		initUI();
		initializeNeuralNetwork();
	}

	private void initUI() {
		board = new Board();
		add(board);

		setTitle("Space Invaders");
		setSize(Commons.BOARD_WIDTH, Commons.BOARD_HEIGHT);

		setDefaultCloseOperation(EXIT_ON_CLOSE);
		setResizable(false);
		setLocationRelativeTo(null);
	}


	public static void showControllerPlaying(GameController controller, long seed) {
		EventQueue.invokeLater(() -> {

			var ex = new SpaceInvaders();
			ex.setController(controller);
			ex.setSeed(seed);
			ex.setVisible(true);
		});
	}

	public static void showAiPlaying(long seed) {
		EventQueue.invokeLater(() -> {

			var ex = new SpaceInvaders();
			ex.setController(new AiController(ex.getNeuralNetwork()));
			ex.setSeed(seed);
			ex.setVisible(true);
		});
	}

	private void initializeNeuralNetwork() {
		int inputDim = Commons.STATE_SIZE;
		int hiddenDim = 100;
		int outputDim = 4;

		SpaceInvadersGeneticAlgorithm geneticAlgorithm = new SpaceInvadersGeneticAlgorithm(inputDim, hiddenDim, outputDim);
		neuralNetwork = geneticAlgorithm.evolve(board);
		board.setController(new AiController(neuralNetwork));
	}

	public void setController(GameController controller) {
		board.setController(controller);
	}

	public void setSeed(long seed) {
		board.setSeed(seed);

	}

	public SpaceInvadersNeuralNetwork getNeuralNetwork() {
		return neuralNetwork;
	}

}
