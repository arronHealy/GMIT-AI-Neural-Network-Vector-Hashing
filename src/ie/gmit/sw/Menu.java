package ie.gmit.sw;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;

public class Menu {

	private int nSizesLength;

	private int[] nSizes;
	
	private int[] functions;

	private int vectorSize;

	private boolean nGramSet = false, vectorSet = false, hiddenLayerSet = false, epochsSet = false, fileSet = false;

	private boolean dataReady = false, networkTrained = false;

	private int numHiddenLayers;

	private int epochs;

	private int consoleChoice;

	private String filePath;

	private File languageFile;

	private Language[] langs;

	private BasicNetwork loadedNetwork;

	private Future<MLData> langVector;

	private MLData languageVector;

	private ExecutorService es;

	private Scanner console;
	
	/*
	 * Menu constructor sets up console and thread executor
	 */
	public Menu() {
		es = Executors.newSingleThreadExecutor();
		console = new Scanner(System.in);
		langs = Language.values();
	}

	/*
	 * The go method provides the user with the menu and prompts for input
	 */
	public void go() {
		System.out.println("=======================================================");
		System.out.println("A Language Detection Neural Network with Vector Hashing");
		System.out.println("=======================================================");

		do {
			System.out.println("\nPlease enter a number from the following choices:");
			System.out.println("1. Set N-Gram Size");
			System.out.println("2. Set Vector Size");
			System.out.println("3. Set Number of Hidden Layers in Neural Network");
			System.out.println("4. Set Number of Epochs");
			System.out.println("5. Set File path to Detect Language");
			System.out.println("6. Prepare Training Data");
			System.out.println("7. Train Neural Network");
			System.out.println("8. Detect Language from File");
			System.out.println("9. Exit Application");

			System.out.print("Enter choice: ");
			consoleChoice = console.nextInt();

			switch (consoleChoice) {
			case 1:
				nSizes = setNgramSize();
				break;

			case 2:
				setVectorSize();
				break;

			case 3:
				setHiddenLayerSize();
				setFunctions();
				break;

			case 4:
				setNumberEpochs();
				break;

			case 5:
				setDetectFilePath();
				break;

			case 6:
				prepareData();
				break;

			case 7:
				trainNetwork();
				break;

			case 8:
				detectLanguage();
				break;

			case 9:
				break;

			default:
				System.out.println("Invalid Menu option Entered! Please Try again.");
			}

		} while (consoleChoice != 9);
		
		// shutdown console and threads before exit
		console.close();
		es.shutdown();

	} // go
	
	/*
	 * private method that sets up the n-gram sizes to be used when processing language data
	 */

	private int[] setNgramSize() {
		
		nGramSet = false;
		int nSizes[];
		int nSize;

		do {
			System.out.print("\nEnter N-Gram Size Array length (Recommended Size -> 2): ");
			nSizesLength = console.nextInt();

		} while (nSizesLength < 1);

		nSizes = new int[nSizesLength];

		for (int i = 0; i < nSizesLength; i++) {

			do {
				System.out.print("\nRecommended Sizes (1,2,3,4,5)");
				System.out.print("\nEnter N-Gram Size " + (i + 1) + ": ");
				nSize = console.nextInt();
			} while (nSize < 1);

			nSizes[i] = nSize;

		}

		nGramSet = true;

		return nSizes;
	}

	/*
	 * private method prompts and sets the vector size for language processing
	 */
	private void setVectorSize() {
		vectorSet = false;
		
		do {
			System.out.print("\nEnter Vector size (Recommended Size -> 315): ");
			vectorSize = console.nextInt();

		} while (vectorSize < 1);

		vectorSet = true;
	}
	
	/*
	 * private method prompts for the number of hidden layers to be set in the neural network topology
	 */

	private void setHiddenLayerSize() {
		hiddenLayerSet = false;
		
		do {
			System.out.print("\nEnter Number of Hidden Layers (Recommended Size -> 1): ");
			numHiddenLayers = console.nextInt();

		} while (numHiddenLayers < 0);

		hiddenLayerSet = true;
	}
	
	/*
	 * private method prompts user for the number of epochs the network should be trained for.
	 */

	private void setNumberEpochs() {
		epochsSet = false;
		
		do {
			System.out.print("\nEnter Number of Epochs (Recommended Size -> 5-6): ");
			epochs = console.nextInt();

		} while (epochs < 1);

		epochsSet = true;
	}
	
	/*
	 * private method prompts user for a file path that will be used to predict a given language.
	 */

	private void setDetectFilePath() {
		fileSet = false;
		
		do {
			System.out.print("\nEnter File Path to be processed: ");
			filePath = console.next();

			languageFile = new File(filePath);

		} while (!languageFile.exists() || !languageFile.isFile());

		fileSet = true;
	}
	
	/*
	 * 
	 * private method will check in case user hasn't set desired variables and if not calls methods to set variables.
	 * after variables are set vector processor class is created and prepares training data file
	 */

	private void prepareData() {
		if (!nGramSet) {
			nSizes = setNgramSize();
		}

		if (!vectorSet) {
			setVectorSize();
		}

		try {
			new VectorProcessor(nSizes, vectorSize).go();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		dataReady = true;

	}
	
	/*
	 * 
	 * private method will check that data file has been prepared and if not will return.
	 * if data prepared but desired variables not set function will call methods to set variables.
	 * after all variables set, the neural network will be set up and trained and save network to a file.
	 */

	private void trainNetwork() {
		
		networkTrained = false;
		
		if (!dataReady) {
			System.out.println("Please Prepare the Training Data before Attepmting this step!");
			return;
		}

		if (!hiddenLayerSet) {
			setHiddenLayerSize();
			setFunctions();
		}

		if (!epochsSet) {
			setNumberEpochs();
		}

		new NeuralNetwork(vectorSize, langs.length, numHiddenLayers, epochs, functions).trainAndSave();

		networkTrained = true;
	}
	
	/*
	 * private function will prompt user for a number to represent the type of activation function
	 * they wish to use in each hidden layer of their neural network.
	 */
	
	private void setFunctions()
	{
		functions = new int[numHiddenLayers];
		
		for(int i = 0; i < numHiddenLayers; i++)
		{
			do
			{
				System.out.println("\nFor Hidden Layer " + (i + 1) + ". Select the Activation function you wish to use:");
				System.out.println("1. Activation Relu");
				System.out.println("2. Activation Sigmoid");
				System.out.println("3. Activation TanH");
				System.out.println("4. Activation Log");
				System.out.print("\nEnter choice (Recommended -> TanH): ");
				
				consoleChoice = console.nextInt();
				
				functions[i] = consoleChoice;
				
			}while(consoleChoice < 1 || consoleChoice > 4);
		}
	}
	
	/*
	 * 
	 * private method will check that network has been trained and if not will return.
	 * if network trained then user will be propmted for a file path.
	 * after file has been set, the saved neural network will be loaded
	 * and the language file to detect will be processed in the same way as the training data
	 * the loaded network will then make a prediction based upon the data processed and
	 * predict the given language.
	 */

	private void detectLanguage() {

		if (!networkTrained) {
			System.out.println(
					"The Neural Network has not been trained! Please Train the Network before attempting this step.");
			return;
		}

		if(!fileSet) {
			setDetectFilePath();
		}
		
		loadedNetwork = Utilities.loadNeuralNetwork("./test.nn");

		langVector = es.submit(new FileProcessor(vectorSize, nSizes, languageFile));

		try {
			languageVector = langVector.get();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		int index = 0;
		double max = 0;
		
		MLData output = loadedNetwork.compute(languageVector);
		
		max = output.getData(0);
		
		for (int i = 1; i < output.getData().length; i++) {
			if (max < output.getData(i)) {
				max = output.getData(i);
				index = i;
			}
		}
		
		System.out.println("Predicted Language is -> " + langs[index]);
	}

}
