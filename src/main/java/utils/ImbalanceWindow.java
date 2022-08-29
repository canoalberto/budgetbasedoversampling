package utils;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

public class ImbalanceWindow {

    Instances window;
    int[] classProportions;
    int windowSize;
    int windowOccupancy;

    public ImbalanceWindow (int windowSize, int numberOfClasses){

        this.window = null;
        this.classProportions = new int[numberOfClasses];
        this.windowSize = windowSize;
        this.windowOccupancy = 0;

    }

    public void update(Instance instance){
        if (this.window == null){
            this.window = new Instances(instance.dataset());
        }

        if (this.window.size() == this.windowSize){
            Instance toBeDeleted = this.window.get(0);
            this.classProportions[(int) toBeDeleted.classValue()]--;
            this.window.delete(0);
        }

        this.window.add(instance);
        this.classProportions[(int) instance.classValue()]++;
    }

    public int[] getClassProportions(){
        return  this.classProportions;
    }


}
