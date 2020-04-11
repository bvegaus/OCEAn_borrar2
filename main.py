import pandas as pd
import numpy as np
import genetic_algorithm as genetic
from sklearn.metrics import accuracy_score, mean_absolute_error
import sys

def get_predictions(chromosome, M, C_test):
    """
        Given a chromosome and a prediction matrix, this function calculates the predicted label for all the instances
        of the dataset
    """    
    y_real = C_test.real
    y_test = list()
    predictions = pd.DataFrame(C_test.drop('real', axis = 1))
    for index, row in predictions.iterrows():
        union = list(zip(row.tolist(), chromosome))
        class_sum = dict()
        for k, v in union:
            class_sum[k] = class_sum.get(k,0)+v
        
        predicted = sorted(class_sum.items(), key = lambda x:x[1], reverse = True)[0][0]
        y_test.append(predicted)
        
    return y_real.tolist(), y_test

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mze = 1 - accuracy_score(y_true, y_pred)
    return mae, mze
    



def main():
    ##Variable Initialization


    N = 100
    n_children = 4

    parents_number = 2 #num of parents used in the mutation
    sel_pressure = 10 #How many individuals are used in the tournament selection
    mu_pressure = 0.4 #Probability of mutation
    G = 100 #Number of generations (stop criteria)

    results = pd.read_excel('./results/predictions/predictions_' + sys.argv[1] + '_train_test.xlsx', sheet_name = "E0")
    C_train = results.drop('real', axis = 1)
    columns = [np.str(col) for col in C_train.columns]
    columns.append("fitness")
    chromosomes_data = []


    M = pd.read_excel('./data/cost_matrices.xlsx' , sheetname = np.str(len(results['real'].unique())) + 'matriz') ##An example of cost matrix file is attached in the repository
    print(M)
    ##Train the genetic algorithm for each of the cross validation steps
    for i in range(10):
        sheet = "E"+np.str(i)
        results = pd.read_excel('./results/predictions/predictions_' + sys.argv[1] + '_train_test.xlsx', sheet_name = sheet)
        l_real = results['real']
        C_train = results.drop('real', axis = 1)
        t = len(C_train.columns)        

        parent, fitness = genetic.genetic_algorithm(C_train, l_real, M, sel_pressure, t, parents_number, n_children, N, mu_pressure, G)
        print("\nFitness in iteration %d:\n%d"%(i, fitness))    
        row = [elem for elem in parent]
        row.append(fitness)
        chromosomes_data.append(row)        
        
    chromosomes = pd.DataFrame(chromosomes_data, columns = columns)
    print(chromosomes)
    chromosomes.to_csv('./results/evaluation/chromosomes_'+ sys.argv[1] +'.csv')  

    ## Evaluation
    res = pd.DataFrame(columns = ['MAE', 'MZE'])
    for i in range(10):
        sheet = "T"+np.str(i)
        C_test = pd.read_excel('./results/predictions/predictions_' + sys.argv[1] + '_train_test.xlsx', sheet_name = sheet)
        chromosome = chromosomes.iloc[i][:-1].tolist()
        y_true, y_pred = get_predictions(chromosome, M, C_test)
        
        print("\n\n")
        mae, mze = evaluate(y_true, y_pred)
        res.loc[i] = [mae, mze]

    res.to_csv('./results/evaluation/evaluation_'+ sys.argv[1] +'.csv')    

if __name__ == "__main__":
    main()
