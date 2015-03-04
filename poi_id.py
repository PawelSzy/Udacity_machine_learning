#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi"]
#features_list.append("salary")
features_list.append("bonus")
#features_list.append("exercised_stock_options")
features_list.append("from_this_person_to_poi")
#features_list.append("from_poi_to_this_person")
#features_list.append("shared_receipt_with_poi")
features_list.append("deferral_payments")
#features_list.append("loan_advances")
#features_list.append("restricted_stock_deferred")
#features_list.append("deferred_income")
features_list.append("expenses")
features_list.append("long_term_incentive")


### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### we suggest removing any outliers before proceeding further
if 'TOTAL' in data_dict: del data_dict['TOTAL']
if 'LAY KENNETH L' in data_dict: del data_dict['LAY KENNETH L']
if 'SKILLING JEFFREY K' in data_dict: del data_dict['SKILLING JEFFREY K']


### if you are creating any new features, you might want to do that here
### store to my_dataset for easy export below
my_dataset = data_dict



### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)



### if you are creating new features, could also do that here

print "features_list", features_list

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

# print "features"
# for feature in features:
    # print feature
    
### machine learning goes here!
### please name your classifier clf for easy export below

#feature scaling - normalizacja feature - x'=(x-xmin)/(xmax-xmin) - wartosc - od "0"-"1"
    #create [dict[xmax:, xmin:, ]] for every feature - znajdz dla kazdego feature wartosc max i min
table_max_min = [{'x_max':0, 'x_min':1} for x_dummy in xrange(len(features[0]))]
#print "table_max_min", table_max_min
for insider in features:
    for nr_feature in xrange(len(insider)):
        if insider[nr_feature] > table_max_min[nr_feature]["x_max"]:
            table_max_min[nr_feature]["x_max"] =  insider[nr_feature]
        if insider[nr_feature] < table_max_min[nr_feature]["x_min"]:
            table_max_min[nr_feature]["x_min"] =  insider[nr_feature]
#print "table_max_min\n", table_max_min
    #normalize feature x'=(x-xmin)/(xmax-xmin)
for insider in features:
    for nr_feature in xrange(len(insider)):
        x_val = insider[nr_feature]
        x_max, x_min = table_max_min[nr_feature]["x_max"], table_max_min[nr_feature]["x_min"]
        insider[nr_feature]= (x_val-x_min)/(x_max-x_min)
        
#split labels and features into train set and test set - podziel dane na traning i test
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)
print "features_train", len(features_train)

#train clf - predict outcome - zacznij trenowac dane, oblicz test
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=10000)
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()

print "Start fit"
print "features_train, labels_train", len(features_train), len(labels_train)
clf = clf.fit(features_train, labels_train)
print "fit"
predictions = clf.predict(features_test)


#show metrics (accuracy, recall_score, precision_score) - wyswietl metryki jak dobrze algorytm sobie radzi 
print "\nlen\n predictions", len(predictions)

accuracy = clf.score(features_test, labels_test)

print "accuracy after clf: ", accuracy
#print "predictions", predictions
#print "\nlabels_test", labels_test

from sklearn.metrics import precision_score, recall_score
print "precision_score", precision_score(predictions, labels_test) 
print "recall_score", recall_score(predictions, labels_test)

#show points in graphic, pokaz punkty w grafice
import matplotlib.pyplot
for nr_first_value in xrange(len(features_list)):
    for nr_second_value in xrange(len(features_list)):
        if nr_first_value!=nr_second_value:
            for point in data:
                first_value = point[nr_first_value]
                second_value = point[nr_second_value]

                #change colors
                is_poi = True if point[0]==1 else False     
                color_point = "r" if is_poi==True else "b"    

    
                matplotlib.pyplot.scatter( first_value, second_value, color=color_point, marker="*" )
            # matplotlib.pyplot.xlabel(features_list[nr_first_value])
            # matplotlib.pyplot.ylabel(features_list[nr_second_value])
            # matplotlib.pyplot.show()

### dump your classifier, dataset and features_list so 
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )



