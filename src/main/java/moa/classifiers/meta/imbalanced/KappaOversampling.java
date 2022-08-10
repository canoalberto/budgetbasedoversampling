package moa.classifiers.meta.imbalanced;

import java.util.Arrays;
import java.util.HashMap;

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

    public ClassOption learnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class, "trees.HoeffdingAdaptiveTree");
    
    public IntOption windowSizeOption = new IntOption("windowSize", 'w', "Window Size", 500, 1, Integer.MAX_VALUE);
    
    public IntOption numberOfNeighborsOption = new IntOption("neighborSize", 'k', "Number of Neighbors", 10, 1, Integer.MAX_VALUE);
    
    public IntOption maxInstancesOption = new IntOption("maxInstances", 'i', "Max number of instances to be created", 50, 1, Integer.MAX_VALUE);
    
    public FloatOption safeCuttingOption = new FloatOption("safeCutting", 's', "Number of neighbors to define an instance as safe", 0.5, 0, 1);
    
	public FloatOption theta = new FloatOption("theta", 't', "The time decay factor for class size.", 0.99, 0, 1);
    
    private Classifier classifier;
    private double[] classProportions;
    private NearestNeighbor nn;
	protected WindowKappaClassificationPerformanceEvaluator evaluator;
    
    @Override
    public void resetLearningImpl() {
    	this.classifier = (Classifier) getPreparedClassOption(this.learnerOption);
    	this.classifier.setModelContext(getModelContext());
        this.classifier.prepareForUse();
        this.classProportions = null;
        this.nn = null;
        this.evaluator = new WindowKappaClassificationPerformanceEvaluator();
    }


    @Override
    public double[] getVotesForInstance(Instance instance) {
        return this.classifier.getVotesForInstance(instance);
    }


    @Override
    public void trainOnInstanceImpl(Instance instance) {

        if (this.classProportions == null) {
            this.classProportions = new double[instance.numClasses()];
        }

        if (this.nn == null) {
            this.nn = new NearestNeighbor(new LinearNNSearch(), windowSizeOption.getValue() * instance.numClasses(), instance); // TODO: window size * number classes (too big!)
        }
        
        double[] votes = this.classifier.getVotesForInstance(instance);
        this.evaluator.addResult(new InstanceExample(instance), votes);
        
        this.classifier.trainOnInstance(instance);
        
        this.nn.add(instance);

        // Update class proportions
        for (int i = 0; i < this.classProportions.length; i++) {
            this.classProportions[i] = theta.getValue() * this.classProportions[i] + (1d - theta.getValue()) * ((int) instance.classValue() == i ? 1d : 0d);
        }

        double imbalanceRatio = this.classProportions[(int) instance.classValue()] / this.classProportions[Utils.maxIndex(this.classProportions)];

        // Compute NN search
        Instances neighbors = nn.getNearestNeighbors(instance, this.numberOfNeighborsOption.getValue());
        Integer[] numberInstancesClassNeighborhood = nn.getClassProportions();
        int numberNeighborsSameClass = numberInstancesClassNeighborhood[(int) instance.classValue()]; 

        if (numberNeighborsSameClass == 0) { // rare

            // Do not oversample
        	System.out.println("Instance is rare");

        } else if (numberNeighborsSameClass < this.numberOfNeighborsOption.getValue() * this.safeCuttingOption.getValue()) { // borderline

        	// Do not oversample
        	System.out.println("Instance is overlapping");

        } else { // safe example

        	System.out.println("Instance is safe");
        	
        	double kappa = this.evaluator.getKappa();
        	
            int k = (int) Math.ceil(this.maxInstancesOption.getValue()*(1-imbalanceRatio)*(1-kappa));

            for (int i = 0; i < k; i++) {

                int randomNeighbor = Math.abs(this.classifierRandom.nextInt() % neighbors.numInstances());

                while (neighbors.get(randomNeighbor).classValue() != instance.classValue()) {
                    randomNeighbor = Math.abs(this.classifierRandom.nextInt() % neighbors.numInstances());
                }

                Instances syntheticInstances = generateLineInstances(instance, neighbors.get(randomNeighbor), 1);
                
                System.out.println("Oversampling with " + syntheticInstances.numInstances() + " artificial instances. Having imbalanceRatio " + imbalanceRatio + " and kappa " + kappa);

                for (int j = 0; j < syntheticInstances.numInstances(); j++) {
                    this.classifier.trainOnInstance(syntheticInstances.get(j));
                }
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
        Instances generatedInstances = InstanceUtils.createInstances(instance);

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
    private HashMap<Integer, InstanceWindow> classWindows = new HashMap<Integer, InstanceWindow>();
    private InstanceWindow windowInstances;
    private int windowSize;
    private Instances actualNeighbors;

    public NearestNeighbor(NearestNeighbourSearch nnSearch, int windowSize, Instance instanceTemplate) {
        this.nnSearch = nnSearch;
        this.windowSize = windowSize;
        this.windowInstances = new InstanceWindow(windowSize, instanceTemplate, false);
    }

    public void add(Instance instance) {

        this.windowInstances.add(instance);
        int instClass = (int) instance.classValue();
        if (!classWindows.containsKey(instClass)) {
            classWindows.put(instClass, new InstanceWindow(this.windowSize / instance.numClasses(), instance, false));
        }
        InstanceWindow specificClassWindow = classWindows.get(instClass);
        if (specificClassWindow.getWindowSize() == this.windowSize / instance.numClasses()) {
            this.windowInstances.remove(specificClassWindow.getInstances().instance(0));
        }
        specificClassWindow.add(instance);
    }

    public InstanceWindow getClassWindow(int i) {
        return this.classWindows.get(i);
    }


    public Instances getNearestNeighbors(Instance instance, int k) {
        try {
            this.nnSearch.setInstances(this.windowInstances.getInstances());
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

    public Integer[] getClassProportions() {
        int numberOfClasses = this.windowInstances.getInstances().numClasses();
        Integer[] classProportions = new Integer[numberOfClasses];
        Arrays.fill(classProportions, 0);
        for (int i = 0; i < this.actualNeighbors.size(); i++) {
            classProportions[(int) this.actualNeighbors.get(i).classValue()]++;
        }
        return classProportions;
    }

    public Instance getCentroid() {
        return this.windowInstances.getCentroid();
    }
}

class InstanceWindow {

    private int windowSize;
    private Instances window;
    private Instance centroid;
    private Instance stdDev;
    private Instance sum;
    private Instance sum_squared;

    public InstanceWindow(int windowSize, Instance instanceTemplate, boolean insert) {
        this.windowSize = windowSize;
        this.window = InstanceUtils.createInstances(instanceTemplate);
        this.centroid = instanceTemplate.copy();
        this.stdDev = instanceTemplate.copy();
        this.sum = instanceTemplate.copy();
        this.sum_squared = instanceTemplate.copy();

        if (insert) {
            this.window.add(instanceTemplate.copy());
        } else {
            for (int i = 0; i < this.centroid.numAttributes() - 1; i++) {
                this.centroid.setValue(i, 0);
                this.stdDev.setValue(i, 0);
                this.sum.setValue(i, 0);
                this.sum_squared.setValue(i, 0);
            }
        }
    }

    public int getWindowSize() {
        return this.window.size();
    }

    public void add(Instance instance) {

        this.update(instance);

        if (this.window.size() == windowSize) {
            this.window.delete(0);
        }

        this.window.add(instance);
    }

    public void remove(Instance instance) {
        for (int i = 0; i < this.window.size(); i++) {
            if (this.window.get(i).equals(instance)) {
                this.window.delete(i);
                break;
            }
        }
    }

    private void update(Instance instance) {
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            double newAttributeAverage = this.centroid.value(i);
            double sum = this.sum.value(i);
            double sum_squared = this.sum_squared.value(i);
            double std = 0;


            if (this.window.size() == windowSize) {
                newAttributeAverage = newAttributeAverage * this.window.size() - this.window.get(0).value(i);
                newAttributeAverage = (newAttributeAverage + instance.value(i)) / this.window.size();

                sum = sum - this.window.get(0).value(i) + instance.value(i);
                sum_squared = sum_squared - Math.pow(this.window.get(0).value(i), 2) + Math.pow(instance.value(i), 2);
                std = Math.sqrt((sum_squared / this.window.size()) - Math.pow((sum / this.window.size()), 2));

            } else {
                newAttributeAverage = (newAttributeAverage * this.window.size() + instance.value(i)) / (this.window.size() + 1);

                sum = sum + instance.value(i);
                sum_squared = sum_squared + Math.pow(instance.value(i), 2);
                std = Math.sqrt((sum_squared / (this.window.size() + 1)) - Math.pow((sum / (this.window.size() + 1)), 2));
            }

            this.centroid.setValue(i, newAttributeAverage);
            this.stdDev.setValue(i, std);
        }
    }

    public Instances getInstances() {
        return this.window;
    }

    public Instance getCentroid() {
        return this.centroid;
    }

    public Instance getStdDev() {
        return this.stdDev;
    }
}

class InstanceUtils {
    public static Instances createInstances(Instance templateInstance) {
        Instances newInstances = new Instances(templateInstance.copy().dataset());
        newInstances.delete();
        return newInstances;
    }
}