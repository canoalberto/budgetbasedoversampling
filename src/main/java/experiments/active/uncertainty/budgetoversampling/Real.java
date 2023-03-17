package experiments.active.uncertainty.budgetoversampling;

import java.io.File;

import org.apache.commons.lang3.SystemUtils;

import utils.Utils;

public class Real {

	public static void main(String[] args) throws Exception {
		
		String absolutePath = SystemUtils.IS_OS_UNIX ? "/home/acano/Downloads/activelearning/" : "D:/activelearning/";
		String resultsPath = "results/uncertainty-budgetbasedoversampling/real/";

		String[] datasets = new String[] {
				"activity",
				"connect-4",
				"CovPokElec",
				"covtype",
				"crimes",
				"fars",
				"gas",
				"hypothyroid",
				"kddcup",
				"kr-vs-k",
				"lymph",
				"olympic",
				"poker",
				"sensor",
				"shuttle",
				"tags",
				"thyroid",
				"zoo",
		};
		
		String[] algorithms = new String[] {
				"moa.classifiers.trees.GHVFDT",
				"moa.classifiers.trees.HDVFDT",
				"moa.classifiers.meta.imbalanced.ROSE",
				"moa.classifiers.meta.OzaBagAdwin",
				"moa.classifiers.meta.LeveragingBag",
				"moa.classifiers.meta.AdaptiveRandomForest",
				"moa.classifiers.meta.OOB",
				"moa.classifiers.meta.UOB",
		};

		String[] algorithmsFilename = new String[algorithms.length];

		for(int alg = 0; alg < algorithms.length; alg++)
			algorithmsFilename[alg] = algorithms[alg].replaceAll(" ", "").replaceAll("moa.classifiers.meta.imbalanced.", "").replaceAll("moa.classifiers.meta.", "").replaceAll("moa.classifiers.trees.", "").replaceAll("moa.classifiers.ann.meta.", "").replaceAll("moa.classifiers.active.", "").replaceAll("[()]", "");

		//String[] activeLearningStrategies = new String [] {"FixedUncertainty", "VarUncertainty", "RandVarUncertainty", "SelSampling"};
		String[] activeLearningStrategies = new String [] {"Random"};
		double[] budgets = new double[] {1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001};
		
		// Check partially-complete experiments (if fails before reaching end)
		for(int strategy = 0; strategy < activeLearningStrategies.length; strategy++) {
			for(int budget = 0; budget < budgets.length; budget++) {
				for(int dataset = 0; dataset < datasets.length; dataset++)
				{
					int maxlines = 0;
					int[] lines = new int[algorithms.length];
		
					for(int alg = 0; alg < algorithms.length; alg++)
					{
						String filename = absolutePath+resultsPath + algorithmsFilename[alg] + "-" + datasets[dataset] + "-" + activeLearningStrategies[strategy] + "-" + budgets[budget] + ".csv";
		
						if(new File(filename).exists())
						{
							lines[alg] = Utils.countLines(filename);
							if(lines[alg] > maxlines) maxlines = lines[alg];
						}
					}
		
					for(int alg = 0; alg < algorithms.length; alg++)
					{
						if(lines[alg] != maxlines)
						{
							String filename = absolutePath+resultsPath + algorithmsFilename[alg] + "-" + datasets[dataset] + "-" + activeLearningStrategies[strategy] + "-" + budgets[budget] + ".csv";
							System.out.println("Incomplete file " + lines[alg] + "\t" + maxlines + "\t" + filename);
							//new File(path + filename).delete();
						}
					}
				}
			}
		}
		
		// Executables
		System.out.println("===== Executables =====");
		
		/* Makefile style */
		/* Execute via: nohup make -i -k -j 24 -f run.sh > nohup.out & */
		System.out.print("all: ");
		int seq = 0;
		for(int dataset = 0; dataset < datasets.length; dataset++) {
			for(int alg = 0; alg < algorithms.length; alg++) {
				for(int strategy = 0; strategy < activeLearningStrategies.length; strategy++) {
					for(int budget = 0; budget < budgets.length; budget++) {
						String filename = absolutePath+resultsPath + algorithmsFilename[alg] + "-" + datasets[dataset] + "-" + activeLearningStrategies[strategy] + "-" + budgets[budget] + ".csv";

						if(!new File(filename).exists()) {
							System.out.print("real-" + seq + " ");
						}
						
						seq++;
		}}}}
		System.out.print("\n");
		
		seq = 0;
		for(int dataset = 0; dataset < datasets.length; dataset++) {
			for(int alg = 0; alg < algorithms.length; alg++) {
				for(int strategy = 0; strategy < activeLearningStrategies.length; strategy++) {
					for(int budget = 0; budget < budgets.length; budget++) {
				
						String VMargs = "-Xms8g -Xmx1024g";
						String jarFile = "budgetbasedoversampling-1.0-jar-with-dependencies.jar";
		
						String filename = absolutePath+resultsPath + algorithmsFilename[alg] + "-" + datasets[dataset] + "-" + activeLearningStrategies[strategy] + "-" + budgets[budget] + ".csv";
	
						if(!new File(filename).exists()) {
							System.out.println("real-" + seq + ": ");

							if (activeLearningStrategies[strategy].equals("Random")){
								resultsPath = "results/random-budgetbasedoversampling/semi-synth/";
								System.out.println("\tjava " + VMargs + " -javaagent:sizeofag-1.0.4.jar -cp " + jarFile + " "
										+ "moa.DoTask moa.tasks.meta.ALPrequentialEvaluationTask"
										+ " -e \"(ALMultiClassImbalancedPerformanceEvaluator -w 500)\""
										+ " -s \"(ArffFileStream -f datasets/real/" + datasets[dataset] + ".arff)\""
										+ " -l \"(moa.classifiers.active.ALRandom -l (moa.classifiers.meta.imbalanced.BudgetImbalancedOversampling -l " + algorithms[alg] + ") -b (moa.classifiers.active.budget.FixedBM -b " + budgets[budget] + "))\""
										+ " -f 500"
										+ " -d " + resultsPath + algorithmsFilename[alg] + "-" + datasets[dataset] + "-" + activeLearningStrategies[strategy] + "-" + budgets[budget] + ".csv");


							}else {

								System.out.println("\tjava " + VMargs + " -javaagent:sizeofag-1.0.4.jar -cp " + jarFile + " "
										+ "moa.DoTask moa.tasks.meta.ALPrequentialEvaluationTask"
										+ " -e \"(ALMultiClassImbalancedPerformanceEvaluator -w 500)\""
										+ " -s \"(ArffFileStream -f datasets/real/" + datasets[dataset] + ".arff)\""
										+ " -l \"(moa.classifiers.active.ALUncertainty -l (moa.classifiers.meta.imbalanced.BudgetImbalancedOversampling -l " + algorithms[alg] + ") -d " + activeLearningStrategies[strategy] + " -b " + budgets[budget] + ")\""
										+ " -f 500"
										+ " -d " + resultsPath + algorithmsFilename[alg] + "-" + datasets[dataset] + "-" + activeLearningStrategies[strategy] + "-" + budgets[budget] + ".csv");
							}
						}
						
						seq++;
					}
				}
			}
		}

		/*// Show metrics for results
		System.out.println("===== Results =====");
		for(int strategy = 0; strategy < activeLearningStrategies.length; strategy++) {
			for(int budget = 0; budget < budgets.length; budget++) {
				
				String[] datasetsFilename = datasets.clone();
				
				for(int i = 0; i < datasetsFilename.length; i++)
					datasetsFilename[i] = datasetsFilename[i] + "-" + activeLearningStrategies[strategy] + "-" + budgets[budget];
				
//				Utils.metric("Kappa", "averaged", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
				Utils.metric("PMAUC", "averaged", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
//				Utils.metric("WMAUC", "averaged", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
//				Utils.metric("EWMAUC", "averaged", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
//				Utils.metric("Accuracy", "averaged", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
//				Utils.metric("G-Mean", "averaged", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
		
//				Utils.metric("evaluation time (cpu seconds)", "last", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
//				Utils.metric("model cost (RAM-Hours)", "averaged", absolutePath+resultsPath, algorithmsFilename, datasetsFilename);
			}
		}*/
	}
}