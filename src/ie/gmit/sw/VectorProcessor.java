package ie.gmit.sw;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.io.InputStreamReader;

public class VectorProcessor {

	private int[] nSizes;

	private double[] vector;

	private Language[] langs;

	private FileWriter csvWriter;

	private DecimalFormat formatter = new DecimalFormat("###.###");
	
	
	/*
	 * VectorProcesor constructor will set up variables for n-gram and vector sizes
	 * get all languages to be processed and set up new file writer
	 */
	public VectorProcessor(int[] nSizes, int vectorSize) throws IOException {
		this.nSizes = nSizes;
		this.vector = new double[vectorSize];
		this.langs = Language.values();
		this.csvWriter = new FileWriter("data.csv");
	}
	
	/*
	 * The go method will read the Wili datafile and process each line of language text until there are no more
	 */
	public void go() {
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("./wili-2018-Small-11750-Edited.txt"))));

			String line = null;

			System.out.println("Processing Data! Please wait...");
			while ((line = br.readLine()) != null) {
				process(line);
			}
			System.out.println("Processing Complete! Training Data Ready.");
			csvWriter.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/*
	 * for each line of text process the n-gram and increment a vector index using the hash code % vector.length formula
	 * after processing vector write input data to file along with the language data 
	 */

	private void process(String line) throws IOException {
		String[] record = line.split("@");

		// return if record not valid
		if (record.length > 2) {
			return;
		}

		String text = record[0].toUpperCase();

		String lang = record[1];

		// initialize vector each time
		for (int i = 0; i < vector.length; i++) {
			vector[i] = 0;
		}
		
		// for each n-gram size get it's hash code index % vector.length value and increment vector index
		for(int i = 0; i < nSizes.length; i++)
		{
			
			for (int j= 0; j < text.length() - nSizes[i]; j++) {
				
				int index = text.substring(j, j + nSizes[i]).hashCode() % vector.length;
				
				if(index < 0) {
					continue;
				}
				vector[index]++;
			}
		}

		// normalize vector data between 0 and 1
		vector = Utilities.normalize(vector, 0, 1);

		// write vector values to file
		for (int i = 0; i < vector.length; i++) {

			csvWriter.write(formatter.format(vector[i]) + ",");

		}

		// write language indexes to file
		for (int i = 0; i < langs.length; i++) {
			if (langs[i].toString().equals(lang)) {
				
				if(i == langs.length - 1) {
					csvWriter.write(Double.toString(1.0));
				}
				else {
					csvWriter.write(Double.toString(1.0) + ",");
				}
			} else {
				if(i == langs.length - 1) {
					csvWriter.write(Double.toString(0.0));
				}
				else {
					csvWriter.write(Double.toString(0.0) + ",");
				}
			}
		}
		
		// add line separator
		csvWriter.write(System.lineSeparator());
	}

}
