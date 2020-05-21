package ie.gmit.sw;

import java.io.File;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.buffer.MemoryDataLoader;
import org.encog.ml.data.buffer.codec.CSVDataCODEC;
import org.encog.ml.data.buffer.codec.DataSetCODEC;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;

public class NeuralNetwork {

	/*
	 * *****************************************************************************
	 * ******** NB: READ THE FOLLOWING CAREFULLY AFTER COMPLETING THE TWO LABS ON
	 * ENCOG AND REVIEWING THE LECTURES ON BACKPROPAGATION AND MULTI-LAYER NEURAL
	 * NETWORKS! YOUR SHOULD ALSO RESTRUCTURE THIS CLASS AS IT IS ONLY INTENDED TO
	 * DEMO THE ESSENTIALS TO YOU.
	 * *****************************************************************************
	 * ********
	 * 
	 * The following demonstrates how to configure an Encog Neural Network and train
	 * it using backpropagation from data read from a CSV file. The CSV file should
	 * be structured like a 2D array of doubles with input + output number of
	 * columns. Assuming that the NN has two input neurons and two output neurons,
	 * then the CSV file should be structured like the following:
	 *
	 * -0.385,-0.231,0.0,1.0 
	 * -0.538,-0.538,1.0,0.0 
	 * -0.63,-0.259,1.0,0.0
	 * -0.091,-0.636,0.0,1.0
	 * 
	 * The each row consists of four columns. The first two columns will map to the
	 * input neurons and the last two columns to the output neurons. In the above
	 * example, rows 1 an 4 train the network with features to identify a category
	 * 2. Rows 2 and 3 contain features relating to category 1.
	 * 
	 * You can normalize the data using the Utils class either before or after
	 * writing to or reading from the CSV file.
	 */

	private BasicNetwork network;

	private int inputs;

	private int outputs;

	private int hiddenLayers;

	private int epochs;
	
	private int[] functions;

	public NeuralNetwork(int inputs, int outputs, int hiddenLayers, int epochs, int[] funcs) {
		this.inputs = inputs; // Change this to the number of input neurons
		this.outputs = outputs; // Change this to the number of output neurons
		this.hiddenLayers = hiddenLayers;
		this.epochs = epochs;
		this.functions = funcs;

		// Configure the neural network topology.
		this.network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, inputs)); // You need to figure out the activation
																				// function
		
		// loop for number of hidden layers and set up each layer with activation function
		for (int i = 0; i < hiddenLayers; i++) {
			
			ActivationFunction func = ActivationFactory.getInstance(functions[i]);
			
			network.addLayer(new BasicLayer(func, true, ((inputs + outputs) / 2 + 15), 0.8));
		}

		network.addLayer(new BasicLayer(new ActivationSoftMax(), false, outputs));
		
		network.getStructure().finalizeStructure();
		
		network.reset();
	}
	
	/*
	 * private method starts the training process for the neural network once structure set up
	 */

	public void trainAndSave() {
		// Read the CSV file "data.csv" into memory. Encog expects your CSV file to have
		// input + output number of columns.
		DataSetCODEC dsc = new CSVDataCODEC(new File("./data.csv"), CSVFormat.DECIMAL_POINT, false, inputs, outputs,
				false);
		MemoryDataLoader mdl = new MemoryDataLoader(dsc);
		MLDataSet trainingSet = mdl.external2Memory();

		// used for cross-validation, gonna slice and dice dataset
		FoldedDataSet folded = new FoldedDataSet(trainingSet);

		MLTrain train = new ResilientPropagation(network, folded);

		// pass in training data and do 5 fold cross validation
		CrossValidationKFold cv = new CrossValidationKFold(train, 5);

		// Train the neural network
		System.out.println("Training Network! Please wait...");
		int epoch = 1; // Use this to track the number of epochs
		long startTime = System.currentTimeMillis();

		do {
			cv.iteration();
			epoch++;
		} while (epoch < epochs);
		cv.finishTraining();

		long trainingTime = (System.currentTimeMillis() - startTime) / 1000;
		
		// print out data related to network training
		System.out.println("Input layer size -> " + inputs);
		System.out.println("Hidden layer size -> " + hiddenLayers);
		System.out.println("Output layer size -> " + outputs);
		System.out.println("Info -> Tested. Accuracy=" + getAccuracy(cv.getTraining()));
		System.out.println("Training completed in " + trainingTime + " seconds.");
		System.out.println("Number of epochs -> " + epoch);
		System.out.println("Error rate -> " + cv.getError());

		// save trained network to file
		Utilities.saveNeuralNetwork(network, "./test.nn");

	}
	
	/*
	 * get accuracy method will pass in training data and return the level accuracy the network performs
	 */
	
	private double getAccuracy(MLDataSet data)
	{
		double total = 0;
		double correct = 0;

		for (MLDataPair pair : data) {
			total++;
			MLData output = network.compute(pair.getInput());

			
			double[] computed = output.getData();
			double computedMax = computed[0];
			int computedMaxIndex = 0;

			for (int i = 1; i < computed.length; i++) {
				if (computed[i] > computedMax) {
					computedMax = computed[i];
					computedMaxIndex = i;
				}
			}

			double[] ideal = pair.getIdeal().getData();
			double idealMax = ideal[0];
			int idealMaxIndex = 0;

			for (int i = 1; i < ideal.length; i++) {
				if (ideal[i] > idealMax) {
					idealMax = ideal[i];
					idealMaxIndex = i;
				}
			}

			if (computedMaxIndex == idealMaxIndex) {
				correct++;
			}

		}
		
		return (correct / total) * 100;
	}
}