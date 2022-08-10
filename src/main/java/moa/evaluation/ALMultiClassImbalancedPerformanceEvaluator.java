/*
 *    ALMultiClassImbalancedPerformanceEvaluator.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.evaluation;

import java.util.Arrays;

import com.yahoo.labs.samoa.instances.Instance;

import moa.core.Example;
import moa.core.Measurement;

/**
 * Classification evaluator that updates evaluation results using a sliding
 * window. Only to be used for binary classification problems with unweighted instances.
 * 
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class ALMultiClassImbalancedPerformanceEvaluator extends MultiClassImbalancedPerformanceEvaluator implements ALClassificationPerformanceEvaluator {

	private static final long serialVersionUID = 1L;

    private moa.evaluation.BasicClassificationPerformanceEvaluator.Estimator acquisitionRateEstimator;

    private int acquiredInstances;

   /**
     * Receives the information if a label has been acquired and increases counters.
     *
     * @param trainInst the instance that was previously considered
     * @param labelAcquired bool type which indicates if trainInst 
     *        was acquired by the active learner
     */
	@Override
	public void doLabelAcqReport(Example<Instance> trainInst, int labelAcquired) {
		this.acquisitionRateEstimator.add(labelAcquired);
		this.acquiredInstances += labelAcquired;
	}

   /**
     * Returns absolute number of acquired labels so far.
     */
	public int getAbsNumOfAcqInst(){
		return acquiredInstances;
	}

   /**
     * Returns relative number of acquired labels so far.
     */
	public double getRelNumOfAcqInst(){
		return acquisitionRateEstimator.estimation();//((float) acquiredInstances) / (float) seenInstances;
	}

	
	@Override
    public Measurement[] getPerformanceMeasurements() {
		Measurement[] measurements = super.getPerformanceMeasurements(); 
		measurements  = Arrays.copyOf(measurements, measurements.length + 2);
		measurements[measurements.length - 2] = new Measurement("Abs Number of Label Acquisitions", getAbsNumOfAcqInst());
		measurements[measurements.length - 1] = new Measurement("Rel Number of Label Acquisitions", getRelNumOfAcqInst());
		return measurements;
    }
	
	@Override
	public void reset(int numClasses) {
		super.reset(numClasses);
		acquisitionRateEstimator = new WindowEstimator(widthOption.getValue());
		acquiredInstances = 0;
		
	}
	
	public class WindowEstimator implements moa.evaluation.BasicClassificationPerformanceEvaluator.Estimator {

		private static final long serialVersionUID = 1L;

		protected double[] window;

        protected int posWindow;

        protected int lenWindow;

        protected int SizeWindow;

        protected double sum;

        protected double qtyNaNs;

        public WindowEstimator(int sizeWindow) {
            window = new double[sizeWindow];
            SizeWindow = sizeWindow;
            posWindow = 0;
            lenWindow = 0;
        }

        public void add(double value) {
            double forget = window[posWindow];
            if(!Double.isNaN(forget)){
                sum -= forget;
            }else qtyNaNs--;
            if(!Double.isNaN(value)) {
                sum += value;
            }else qtyNaNs++;
            window[posWindow] = value;
            posWindow++;
            if (posWindow == SizeWindow) {
                posWindow = 0;
            }
            if (lenWindow < SizeWindow) {
                lenWindow++;
            }
        }

        public double estimation(){
            return sum / (lenWindow - qtyNaNs);
        }
    }
}