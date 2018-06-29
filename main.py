#Breast Cancer Detection Using k Nearest Neighbors Algorithm.

#importing the required libraries
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

#k nearest neighbor algorithm
def k_nearest_neighbors(data, predict, k=3):
    if len(data) > k: #if k greater than the dataset then it will give a warning message.
        warnings.warn('K is set to a value less than total voting groups!')

    distances = [] #euclidean distances
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) #calculating distances of nearest neighbors
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]] #returns a list containing the labels upto Kth index
    result = Counter(votes).most_common(1)[0][0] #returns a tuple containing common votes
    confidence = Counter(votes).most_common(1)[0][1] / k #calculating the confidence
    return result, confidence


df = pd.read_csv('breast-cancer-wisconsin.txt') #read the csv dataset into pandas dataframe
df.replace('?',-99999, inplace=True) #replacing the '?' values with -99999 so that the algorithm treats it as outlier.
df.drop(['id'],1,inplace=True)

#converting the data to list of list and datatype
#into float, there might contain string feature so.
full_data = df.astype(float).values.tolist() #converting the dataframe into list having float features.
random.shuffle(full_data)

test_size = 0.2 #size of the test_set
train_set = {2:[],4:[]} #declaring empty train_set dictionary
test_set = {2:[],4:[]}  #declaring empty test_set dictionary
train_data = full_data[:-int(test_size * len(full_data))] #loading the 80% of the features in train_data
test_data = full_data[-int(test_size * len(full_data)):] #loading the remaining of the features in test_data


for i in train_data:  #loading the train_set dictioinary which does not contain the label
    train_set[i[-1]].append(i[:-1])

for i in test_data:   ##loading the test_set dictioinary which does not contain the label
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set: #training the algorithm
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1


print('Accuracy of the algorithm: %.3f'%((correct/total)*100))
print('Confidence score: ',confidence)

#To test the benign and malignant
#benign is represented as 2
#malignant is represented as 4
#testing the model using data points from real world symptoms.
test = []
print('\nEnter symptoms')
clump_thickness= int(input('1. Enter Clump Thickness: '))
test.append(clump_thickness)
unif_cell_size= int(input('2. Enter Uniform Cell Size: '))
test.append(unif_cell_size)
unif_cell_shape= int(input('3. Enter Uniform Cell Shape: '))
test.append(unif_cell_shape)
marg_adhesion= int(input('4. Enter Marginal Adhesion: '))
test.append(marg_adhesion)
single_epith_cell_size= int(input('5. Enter Single Epithelial Cell Size: '))
test.append(single_epith_cell_size)
bare_nuclei = int(input('6. Enter the Bare nuclei: '))
test.append(bare_nuclei)
bland_chrom = int(input('7. Enter the Bland chromatin: '))
test.append(bland_chrom)
norm_nucleoli = int(input('8. Enter the Normal Nucleoli: '))
test.append(norm_nucleoli)
mitoses = int(input('9. Enter the Mitoses: '))
test.append(mitoses)

result, certainty = k_nearest_neighbors(train_set, test, k=5)
if result == 2:
    print("\nThe patient is not malignant")
else:
    print("\nThe patient is malignant")
