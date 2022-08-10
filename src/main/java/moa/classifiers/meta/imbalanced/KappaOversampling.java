package moa.classifiers.meta.imbalanced;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.Utils;
import moa.evaluation.WindowKappaClassificationPerformanceEvaluator;
import moa.options.ClassOption;

public class KappaOversampling extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class, "moa.classifiers.trees.HoeffdingAdaptiveTree");
    
    public IntOption windowSizeOption = new IntOption("windowSize", 'w', "Window Size", 500, 1, Integer.MAX_VALUE);
    
    public IntOption numberOfNeighborsOption = new IntOption("neighborSize", 'k', "Number of Neighbors", 10, 1, Integer.MAX_VALUE);
    
    public IntOption maxInstancesOption = new IntOption("maxInstances", 'i', "Max number of instances to be created", 50, 1, Integer.MAX_VALUE);
    
    public FloatOption safeCuttingOption = new FloatOption("safeCutting", 's', "Number of neighbors to define an instance as safe", 0.5, 0, 1);
    
	public FloatOption theta = new FloatOption("theta", 't', "The time decay factor for class size.", 0.99, 0, 1);
    
    private Classifier classifier;
    private double[] numberInstancesClassPrequential;
    private NearestNeighbor nn;
	protected WindowKappaClassificationPerformanceEvaluator evaluator;
    
    @Override
    public void resetLearningImpl() {
    	this.classifier = (Classifier) getPreparedClassOption(this.learnerOption);
    	this.classifier.setModelContext(getModelContext());
        this.classifier.prepareForUse();
        this.numberInstancesClassPrequential = null;
        this.nn = null;
        this.evaluator = new WindowKappaClassificationPerformanceEvaluator();
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        return this.classifier.getVotesForInstance(instance);
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {

        if (this.numberInstancesClassPrequential == null) {
            this.numberInstancesClassPrequential = new double[instance.numClasses()];
        }

        if (this.nn == null) {
            this.nn = new NearestNeighbor(new LinearNNSearch(), windowSizeOption.getValue(), instance);
        }
        
        double[] votes = this.classifier.getVotesForInstance(instance);
        this.evaluator.addResult(new InstanceExample(instance), votes);
        
        this.classifier.trainOnInstance(instance);
        
        this.nn.add(instance);

        // Update class proportions
        for (int i = 0; i < this.numberInstancesClassPrequential.length; i++) {
            this.numberInstancesClassPrequential[i] = theta.getValue() * this.numberInstancesClassPrequential[i] + (1d - theta.getValue()) * ((int) instance.classValue() == i ? 1d : 0d);
        }

        double imbalanceRatio = this.numberInstancesClassPrequential[(int) instance.classValue()] / this.numberInstancesClassPrequential[Utils.maxIndex(this.numberInstancesClassPrequential)];
    	double kappa = this.evaluator.getKappa();

        // Compute NN search
        Instances neighbors = nn.getNearestNeighbors(instance, this.numberOfNeighborsOption.getValue());
        int[] numberInstancesClassNeighborhood = nn.getNumberInstancesClassNeighbors();
        int numberNeighborsSameClass = numberInstancesClassNeighborhood[(int) instance.classValue()]; 

        if (numberNeighborsSameClass == 0) { // rare
        	System.out.println("Instance type is rare");
        } else if (numberNeighborsSameClass < this.numberOfNeighborsOption.getValue() * this.safeCuttingOption.getValue()) { // borderline
        	System.out.println("Instance type is overlapping");
        } else { // safe example
        	System.out.println("Instance type is safe");
        	oversampleandtrain(instance, neighbors, imbalanceRatio, kappa);
        }
    }
    
    private void oversampleandtrain(Instance instance, Instances neighbors, double imbalanceRatio, double kappa) {
    	
        int k = (int) Math.ceil(this.maxInstancesOption.getValue()*(1-imbalanceRatio)*(1-kappa));
        
        if (k > 0)
        	System.out.println("Oversampling with " + k + " artificial instances. Having imbalanceRatio " + imbalanceRatio + " and kappa " + kappa);
        else
        	System.out.println("NOT oversampling. Having imbalanceRatio " + imbalanceRatio + " and kappa " + kappa);

        for (int i = 0; i < k; i++) {

            int randomNeighbor = Math.abs(this.classifierRandom.nextInt() % neighbors.numInstances());

            while (neighbors.get(randomNeighbor).classValue() != instance.classValue()) {
                randomNeighbor = Math.abs(this.classifierRandom.nextInt() % neighbors.numInstances());	// TODO: what if there's no neighbor of the intended class? (it could be only "self").
            }

            Instances syntheticInstances = generateLineInstances(instance, neighbors.get(randomNeighbor), 1);
            
            System.out.println("Generating using neighbor " + neighbors.get(randomNeighbor).hashCode() + " current " + instance.hashCode());

            for (int j = 0; j < syntheticInstances.numInstances(); j++) {
                this.classifier.trainOnInstance(syntheticInstances.get(j));
            }
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder stringBuilder, int i) {
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
    
    private Instances generateLineInstances(Instance instance, Instance neighbor, int numGenerations) {
        Instances generatedInstances = new Instances(instance.dataset(), 0);

        for (int i = 0; i < numGenerations; i++) {
            Instance newInstance = instance.copy();
            newInstance.setClassValue(instance.classValue());
            
            for (int j = 0; j < instance.numAttributes() - 1; j++) {
                if (instance.attribute(j).isNumeric()) {
                    newInstance.setValue(j, instance.value(j) + this.classifierRandom.nextDouble() * (neighbor.value(j) - instance.value(j)));
                } else {
                    newInstance.setValue(j, this.classifierRandom.nextBoolean() ? instance.value(j) : neighbor.value(j));
                }
            }
            
            generatedInstances.add(newInstance);
        }
        
        return generatedInstances;
    }
}

class NearestNeighbor {

    private NearestNeighbourSearch nnSearch;
    private Instances windowInstances;
    private Instances actualNeighbors;
    private int windowSize;

    public NearestNeighbor(NearestNeighbourSearch nnSearch, int windowSize, Instance instanceTemplate) {
        this.nnSearch = nnSearch;
        this.windowSize = windowSize;
        this.windowInstances = new Instances(instanceTemplate.dataset(), 0);
    }

    public void add(Instance instance) {

        this.windowInstances.add(instance);
        
        if (this.windowInstances.size() > this.windowSize) {
            this.windowInstances.delete(0);
        }
    }

    public Instances getNearestNeighbors(Instance instance, int k) {
        try {
            this.nnSearch.setInstances(this.windowInstances);
            this.actualNeighbors = clean(this.nnSearch.kNearestNeighbours(instance, k), k);
            return this.actualNeighbors;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private Instances clean(Instances nearestNeighbors, double k) {
        if (nearestNeighbors == null) return null;

        int nnSize = nearestNeighbors.size();
        if (nnSize > k) {
            for (int i = 0; i < nnSize - k; i++) nearestNeighbors.delete(nnSize - i - 1);
        }

        return nearestNeighbors;
    }

    public int[] getNumberInstancesClassNeighbors() {
        int numberOfClasses = this.windowInstances.numClasses();
        int[] classProportions = new int[numberOfClasses];
        for (int i = 0; i < this.actualNeighbors.size(); i++) {
            classProportions[(int) this.actualNeighbors.get(i).classValue()]++;
        }
        return classProportions;
    }
}