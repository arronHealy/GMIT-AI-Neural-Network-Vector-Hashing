package ie.gmit.sw;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.concurrent.Callable;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class FileProcessor implements Callable<MLData> {
	
	private int[] nSizes;
	
	private double[] vector;
	
	private File inFile;
	
	/*
	 * set up inital variables
	 */
	public FileProcessor(int vectorSize, int[] nSizes, File inFile) {
		this.vector = new double[vectorSize];
		this.nSizes = nSizes;
		this.inFile = inFile;
	}

	/*
	 * call method processes processes vector data and returns an MLData object
	 */
	@Override
	public MLData call() throws Exception {
		
		// initialize vector
		for (int i = 0; i < vector.length; i++) {
			vector[i] = 0;
		}
		
		try {
			BufferedReader br = new BufferedReader(
					new InputStreamReader(new FileInputStream(inFile)));

			String line = null;

			System.out.println("Processing Data! Please wait...");
			while ((line = br.readLine()) != null) {
				
				for(int i = 0; i < nSizes.length; i++)
				{
					for (int j = 0; j < line.length() - nSizes[i]; j++) {
						
						int index = line.substring(j, j + nSizes[i]).hashCode() % vector.length;
						
						if(index < 0) {
							continue;
						}
						
						vector[index]++;
					}
				}

			}

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		// normalize vetor data before returning
		vector = Utilities.normalize(vector, 0, 1);
		
		return new BasicMLData(vector);
	}

}
