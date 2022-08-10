package experiments.active;

import moa.classifiers.active.ALUncertainty;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.evaluation.ALMultiClassImbalancedPerformanceEvaluator;
import moa.streams.ArffFileStream;

public class Debug {

	public static void main(String[] args) throws Exception
	{
		ArffFileStream stream = new ArffFileStream("datasets/multi-class/shuttle.arff", -1);
		stream.prepareForUse();
		
		ALUncertainty activelearning = new ALUncertainty();
		
		activelearning.baseLearnerOption.setValueViaCLIString("moa.classifiers.trees.HoeffdingAdaptiveTree");
//		activelearning.baseLearnerOption.setValueViaCLIString("moa.classifiers.meta.imbalanced.KappaOversampling -l moa.classifiers.trees.HoeffdingAdaptiveTree");
		
		activelearning.activeLearningStrategyOption.setValueViaCLIString("RandVarUncertainty");
		activelearning.budgetOption.setValue(0.1);

		activelearning.prepareForUse();
		activelearning.resetLearning();
		activelearning.setModelContext(stream.getHeader());

		int numberInstances = 0;
		
		ALMultiClassImbalancedPerformanceEvaluator evaluator = new ALMultiClassImbalancedPerformanceEvaluator();

		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
		
		while (stream.hasMoreInstances())
		{
			InstanceExample instance = stream.nextInstance();
			
			evaluator.addResult(instance, activelearning.getVotesForInstance(instance));
			
			//System.out.println(instance.getData().classValue() + "\t" + Utils.maxIndex(learner.getVotesForInstance(instance)));
			
			activelearning.trainOnInstance(instance);
			
        	evaluator.doLabelAcqReport(instance, activelearning.getLastLabelAcqReport());

			numberInstances++;
		}

		double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);

		System.out.println(numberInstances + " instances processed in " + time + " seconds");
		
		for(int i = 0; i < evaluator.getPerformanceMeasurements().length; i++)
			System.out.println(evaluator.getPerformanceMeasurements()[i].getName() + "\t" + evaluator.getPerformanceMeasurements()[i].getValue());
	}
}