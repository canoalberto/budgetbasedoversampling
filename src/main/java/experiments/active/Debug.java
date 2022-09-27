package experiments.active;

import moa.classifiers.active.ALUncertainty;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.evaluation.ALMultiClassImbalancedPerformanceEvaluator;
import moa.streams.ArffFileStream;
import moa.classifiers.active.budget.FixedBM;
import moa.classifiers.active.ALRandom;

import java.util.ArrayList;

public class Debug {

	public static void main(String[] args) throws Exception
	{
		ArffFileStream stream = new ArffFileStream("datasets/semi-synth/DJ30-D1.arff", -1);
		stream.prepareForUse();
		
		//ALUncertainty activelearning = new ALUncertainty();

		ALRandom activelearning = new ALRandom();

		activelearning.budgetManagerOption.setValueViaCLIString("moa.classifiers.active.budget.FixedBM -b 0.2");
		
		activelearning.baseLearnerOption.setValueViaCLIString("moa.classifiers.meta.imbalanced.OSAMP");
		//activelearning.baseLearnerOption.setValueViaCLIString("moa.classifiers.meta.imbalanced" +
		//		".KappaImbOversampling " +
		//		"-l moa.classifiers.meta.imbalanced.OSAMP -b 0.2 -w 200 -f 1");
		
		//activelearning.activeLearningStrategyOption.setValueViaCLIString("RandVarUncertainty");
		//activelearning.budgetOption.setValue(0.2);

		activelearning.prepareForUse();
		activelearning.resetLearning();
		activelearning.setModelContext(stream.getHeader());

		int numberInstances = 0;
		
		ALMultiClassImbalancedPerformanceEvaluator evaluator = new ALMultiClassImbalancedPerformanceEvaluator();
		int eval_size = evaluator.widthOption.getValue();

		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

		double avg_pmauc = 0;
		double avg_kappa = 0;
		int n_windows = 0;

		while (stream.hasMoreInstances())
		{
			InstanceExample instance = stream.nextInstance();
			
			evaluator.addResult(instance, activelearning.getVotesForInstance(instance));
			
			//System.out.println(instance.getData().classValue() + "\t" + Utils.maxIndex(learner.getVotesForInstance(instance)));
			
			activelearning.trainOnInstance(instance);
        	evaluator.doLabelAcqReport(instance, activelearning.getLastLabelAcqReport());


			if (numberInstances%eval_size == 0){
				//System.out.println(evaluator.getPerformanceMeasurements()[1].getName() + "\t" + evaluator
				// .getPerformanceMeasurements()[1].getValue());

				avg_pmauc = avg_pmauc + evaluator.getPerformanceMeasurements()[1].getValue();
				if (evaluator.getPerformanceMeasurements()[5].getValue()>0) {
					avg_kappa = avg_kappa + evaluator.getPerformanceMeasurements()[5].getValue();
				}

				n_windows++;
			}

			numberInstances++;
		}

		System.out.println("AVG PMAUC \t" + avg_pmauc/n_windows);
		System.out.println("AVG KAPPA \t" + avg_kappa/n_windows);

		double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);

		System.out.println(numberInstances + " instances processed in " + time + " seconds");
		
		for(int i = 0; i < evaluator.getPerformanceMeasurements().length; i++)
			System.out.println(evaluator.getPerformanceMeasurements()[i].getName() + "\t" + evaluator.getPerformanceMeasurements()[i].getValue());
	}
}