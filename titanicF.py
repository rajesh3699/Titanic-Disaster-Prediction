import numpy as np
import pandas as pd

#read input data using pandas
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
print("First we shall see how the input data looks:\n")

#getting a look at the data
print(train.head())
print("Next we can observe how much of the data is present and missing:\n")
print(train.info())
print("Let us first fill in Age, Fair and port of Embarkation according to it's means:")

#making array copies of train and test
train_arr=[train]
test_arr=[test]

#finding mean of age
Age_mean = train['Age'].mean()
Age_mean_test = test['Age'].mean()

#setting null values to the mean of age
for dataset in train_arr:
    dataset['Age'] = dataset['Age'].fillna(Age_mean)
    dataset['Age'] = dataset['Age'].astype(int)
for dataset1 in test_arr:
    dataset1['Age'] = dataset1['Age'].fillna(Age_mean_test)
    dataset1['Age'] = dataset1['Age'].astype(int)

#finding mean of fare
Fare_mean = train['Fare'].mean()
Fare_mean_test = test['Fare'].mean()

#setting null values to the mean of fare
for dataset in train_arr:
    dataset['Fare'] = dataset['Fare'].fillna(Fare_mean)
    dataset['Fare'] = dataset['Fare'].astype(int)
for dataset1 in test_arr:
    dataset1['Fare'] = dataset1['Fare'].fillna(Fare_mean_test)
    dataset1['Fare'] = dataset1['Fare'].astype(int)

#setting null values to S as it is the most common port of embarkation
for dataset in train_arr:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in test_arr:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print("Now we have filled in the null values for the fields which contain a majority of filled values:\n")
print(train.head())
print(train.info())

print("We shall now assign integer values to the gender and the port of embarkation")

#setting male to 0 and female to 1
train['Sex'] = train['Sex'].map({"male": 0, "female": 1})
test['Sex'] = test['Sex'].map({"male": 0, "female": 1})

#setting ports of embarkation to 0, 1 and 2
train['Embarked'] = train['Embarked'].map({"S": 0, "C": 1, "Q": 2})
test['Embarked'] = test['Embarked'].map({"S": 0, "C": 1, "Q": 2})

for dataset in train_arr:
     dataset['Embarked'] = dataset['Embarked'].astype(int)
for dataset in test_arr:
     dataset['Embarked'] = dataset['Embarked'].astype(int)

print(train.head())
print(train.info())

print("It is clear that passengerID, ticket, cabin and name cannot play a role in determining the survival of a passenger hence we will remove those fields:")

#dropping passengerID
train = train.drop(['PassengerId'], axis=1)
passid = test['PassengerId']
test = test.drop(['PassengerId'], axis=1)

#dropping ticket
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

#dropping cabin
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

#dropping name
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

#drop survived column from X(training data) and create y(training outputs)
X = train.drop('Survived', axis=1)
y = train['Survived']

#reshaping of the array y from (891,) to (891,1)
yt = np.zeros((X.shape[0],1))
for i in range(y.shape[0]):
    yt[i]= y[i]

#taking transpose before upcoming functions used for training
X=X.T
yt=yt.T
test=test.T

#sigmoid function used as the activation function(convert hypothesis to number between 0 and 1)
def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    return A

#hypothesis function used to obtain the output using current weights, current input and current bias
def hypo(x,w,b):
    h=np.dot(w.T,x)+b
    h=sigmoid(h)
    return h

#The gradient dunction used to calculate the small step required to go in the direction of a more correct output
#according to the training data and output provided
def grad(x,y,w,b):
    m=x.shape[1]
    A=hypo(x,w,b)
    dw = np.dot(x,(A-y).T)/m
    db=np.sum(A-y,axis=1,keepdims=True)
    return dw, db

print("we can now begin to train our weights and bias using training data")
#the actual steps taken in the direction as provided by the gradient function taken 100000 times
w=np.zeros((X.shape[0],1))
b=0
for i in range(100000):
    dw,db = grad(X,yt,w,b)
    w = w- dw*0.001
    b = b- db*0.001
    print("iteration ",i,"/100000")

#now our weights i.e. w and bias i.e. b is said to be "trained". We can now find hypothesis on our test data.
hp_test = hypo(test,w,b)

#hypothesis gives number between 0 and 1. Therefor hypo - 0.5 gives num between -0.5 to 0.5 which, on ceiling, gives us
#either 0 or 1 i.e. output of our test set
y_p_t = np.ceil(hp_test-0.5)
y_p_t=y_p_t.astype(int)

#reshaping outputs from (418,1) to (418,)
y_p_t=y_p_t.T
y_p_t=y_p_t.reshape((418,))

#Our task is now complete and we have used the trained weights and biases on our test set to calculate outputs
#We will now form an output file similar to gender_submission.csv
output = pd.DataFrame({'PassengerId': passid, 'Survived': y_p_t})
output.to_csv('my_submission.csv', index=False)
print("\n")
print("Predicted Data:\n")
print(output)