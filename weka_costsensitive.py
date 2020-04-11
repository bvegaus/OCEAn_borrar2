import traceback
import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import SingleClassifierEnhancer, Classifier, Evaluation
from weka.core.dataset import Instances
from weka.filters import Filter
import pandas as pd
import numpy as np
import time
import sys
import os


def classification(data, train, test, num_clases):
        baseClassifiers_list = ["weka.classifiers.bayes.NaiveBayes", "weka.classifiers.functions.MultilayerPerceptron",
                            "weka.classifiers.functions.SMO","weka.classifiers.lazy.IBk", "weka.classifiers.lazy.KStar", "weka.classifiers.meta.AdaBoostM1",
                            "weka.classifiers.meta.Bagging", "weka.classifiers.meta.LogitBoost", "weka.classifiers.trees.J48", "weka.classifiers.trees.DecisionStump",
                            "weka.classifiers.trees.LMT", "weka.classifiers.trees.RandomForest", "weka.classifiers.trees.REPTree", "weka.classifiers.rules.PART",
                            "weka.classifiers.rules.JRip", "weka.classifiers.functions.Logistic", "weka.classifiers.meta.ClassificationViaRegression", 
                            "weka.classifiers.bayes.BayesNet"]
        results_train = pd.DataFrame()
        results_test = pd.DataFrame()

        cost_matrix_list =  [
        "[]", 
        "[0]", 
        "[0.0 1.0; 1.0 0.0]", 
        "[0.0 1.0 2.0; 1.0 0.0 1.0; 2.0 1.0 0.0]", 
        "[0.0 1.0 2.0 3.0; 1.0 0.0 1.0 2.0; 2.0 1.0 0.0 1.0; 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0; 1.0 0.0 1.0 2.0 3.0; 2.0 1.0 0.0 1.0 2.0; 3.0 2.0 1.0 0.0 1.0; 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0; 1.0 0.0 1.0 2.0 3.0 4.0; 2.0 1.0 0.0 1.0 2.0 3.0; 3.0 2.0 1.0 0.0 1.0 2.0; 4.0 3.0 2.0 1.0 0.0 1.0; 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 9.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0]"  ]



        real_train = [] # the real label of the dataset
        for i in range(train.num_instances):
            real_train.append(train.get_instance(i).values[(train.num_attributes-1)])
        results_train['real'] = real_train

        real_test = [] # the real label of the dataset
        for i in range(test.num_instances):
            real_test.append(test.get_instance(i).values[(test.num_attributes-1)])
        results_test['real'] = real_test
        

        num = 0
        for clas in baseClassifiers_list:
            column = "p"+np.str(num)    
            
            #classifier
            classifier = SingleClassifierEnhancer(
                classname="weka.classifiers.meta.CostSensitiveClassifier",
                options=["-cost-matrix", cost_matrix_list[num_clases], "-M", "-S", "1"])
            base = Classifier(classname=clas)
            classifier.classifier = base    
        
        
            predicted_data_train = None
            predicted_data_test = None

            evaluation = Evaluation(data)
            classifier.build_classifier(train)
            #evaluation.test_model(classifier, train)
            
            # add predictions
            addcls = Filter(
                    classname="weka.filters.supervised.attribute.AddClassification",
                    options=["-classification"])
            
            addcls.set_property("classifier", Classifier.make_copy(classifier))
            addcls.inputformat(train)
            #addcls.filter(train)  # trains the classifier
            pred_train = addcls.filter(train)

            pred_test = addcls.filter(test)


            if predicted_data_train is None:
                predicted_data_train = Instances.template_instances(pred_train, 0)
            for n in range(pred_train.num_instances):
                predicted_data_train.add_instance(pred_train.get_instance(n))


            if predicted_data_test is None:
                predicted_data_test = Instances.template_instances(pred_test, 0)
            for n in range(pred_test.num_instances):
                predicted_data_test.add_instance(pred_test.get_instance(n))



            preds_train = [] #labels predicted for the classifer trained in the iteration
            preds_test = []
    
            
            for i in range(predicted_data_train.num_instances):
                preds_train.append(predicted_data_train.get_instance(i).values[(predicted_data_train.num_attributes-1)])

            for i in range(predicted_data_test.num_instances):
                preds_test.append(predicted_data_test.get_instance(i).values[(predicted_data_test.num_attributes-1)])        

            results_train[column] = preds_train
            results_test[column] = preds_test
            num = num+1
        return results_train, results_test


def main():

    dataset = sys.argv[1]
    #load a dataset
    loader = Loader("weka.core.converters.ArffLoader")
    data = loader.load_file("./data/"+ dataset + ".arff")
    data.class_is_last()

    num_classes = data.class_attribute.num_values






    os.mkdir('resultados_'+ sys.argv[1])
    for random_cv in range(10): #10 CV

            # generate train/test split of randomized data
        train, test = data.train_test_split(75.0, Random(random_cv))
        results_train, results_test  = classification(data, train, test, num_classes)
#        results_test = classification(test, num_classes)
        

              
        
        #Write results in Excel format
        train_name = "./resultados_"+sys.argv[1]+"/resultados_"+sys.argv[1]+"_" + "E"+np.str(random_cv) +".csv"
        test_name = "./resultados_"+sys.argv[1]+"/resultados_"+sys.argv[1]+"_" + "T"+np.str(random_cv)+".csv"

        results_train.to_csv(train_name)
        results_test.to_csv(test_name)

    
if __name__ == "__main__":
    try:
        start_time = time.time()
        jvm.start()
        main()
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception:
        print(traceback.format_exc())
    finally:
        jvm.stop()
