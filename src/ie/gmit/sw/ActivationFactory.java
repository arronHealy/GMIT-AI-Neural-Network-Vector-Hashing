package ie.gmit.sw;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;

public class ActivationFactory {
	
	/*
	 * Activation factory class returns Activation function to be used for a given hidden layer
	 */
	
	public ActivationFactory() {
		// TODO Auto-generated constructor stub
	}
	
	public static ActivationFunction getInstance(int choice) {
		
		switch(choice)
		{
			case 1:
				return new ActivationReLU();
			
			case 2:
				return new ActivationSigmoid();
			
			case 3:
				return new ActivationTANH();
				
			case 4:
				return new ActivationLOG();
		}
		
		return new ActivationReLU();
	}

}
