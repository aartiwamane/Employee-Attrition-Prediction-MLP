#-------------------------------------------------------------
# Build a Deep Learning based employee Attrition Prediction 
# System using MLPClassifier
#-------------------------------------------------------------

#-------------------------------------------------------------
# Tasks to Perfrom :
#-------------------------------------------------------------
# 1. Load the Dataset using pandas
# 2. Display the shape, columns and first five records
# 3. Check for missing values
# 4. Identify numerical and categorical features
# 5. Convert categorical features such as Overtime into numerical representation.
# 6. Convert the tagret Attrition into 0 and 1
# 7. Separate Independent and Dependent Variables 
# 8. Divide the dataset into training and testing data
# 9. Apply appropriate feature scaling
# 10. Design an MLP with at least two hidden layers.
# 11. train the network
# 12. Display the number of iterations required for training and testing
# 13. Calculate training accuracy.
# 14. Calculate testing accuracy.
# 15. Generate a Confusion Matrix
# 16. Plot the loss Curve.
# 17. Create a function : PredictAttrition
# 18. Test the system using at least five new employee records.
# 19. Explain whether the model is suffering from overfitting or underfitting
#-------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

Border= "-"*60
def LoadData():
    print(Border)
    print("Load the Dataset using pandas")
    print(Border)

    df = pd.read_csv("Employee_Attrition.csv")

    return df

def AnalyzeData(df):
    print(Border)
    print("Analyze the the Dataset using pandas")
    print(Border)

    print("Sape of Dataset : ",df.shape)
    print(Border)

    print("Columns Names : ")
    print(df.columns)
    print(Border)

    print("First five records of Dataset : ")
    print(df.head())

def CheckMissing(df):
    print(Border)
    print("Check the Missing Values")
    print(Border)

    print("Missing recored : ")
    print(df.isnull().sum())

def DisplayFeatures(df):
    print(Border)
    print("Display the Features of the Dataset")
    print(Border)

    print("Numerical Features : ")
    print(df.select_dtypes(include=["number"]).columns)

    print("Categorical Features : ")
    print(df.select_dtypes(include=["str","category"]).columns)

def CategoricalToNumerical(df):
    print(Border)
    print("Convert Categorical Data into Numerical")
    print(Border)

    df["OverTime"] = df["OverTime"].map({"Yes" : 1, "No" : 0})
    df['Attrition'] = df['Attrition'].map({"Yes" : 1, "No" : 0})

    return df

def SeparateVariable(df):
    print(Border)
    print("Separate X and Y variable")
    print(Border)

    X = df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears',
       'DistanceFromHome', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime',
       'NumCompaniesWorked', 'TrainingTimesLastYear']]

    Y = df['Attrition']

    return X, Y

def DivideDataset(df,X,Y):
    print(Border)
    print("Split the Dataset into training and testing ")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

    return X_train,X_test,Y_train,Y_test

def FeatureScaling(df,X_train,X_test):
    print(Border)
    print("Scale the Dataset")
    print(Border)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('Scaled Data : ',X_train[:5])

    return X_train,X_test, scaler

def ModelCreation():
    print(Border)
    print("Model Creation ")
    print(Border)

    model = MLPClassifier(
        hidden_layer_sizes=(8,4),
        activation="relu",
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    return model

def ModelTraining(model,X_train,Y_train):
    print(Border)
    print("Model Training ")
    print(Border)

    model = model.fit(X_train,Y_train)

    return model

def ModelTesting(model,X_test,Y_test):
    print(Border)
    print("Model Testing ")
    print(Border)

    Y_pred = model.predict(X_test)
    TestingAccuracy = accuracy_score(Y_pred,Y_test)*100

    print("Testing Accuracy is : ",TestingAccuracy)

    return Y_pred

def ModelTrainingAccuracy(model,X_train,Y_train):
    print(Border)
    print("Display training Accuracy and number of iteration")
    print(Border)

    Y_Pred_train = model.predict(X_train)
    TrainingAccuracy = accuracy_score(Y_train,Y_Pred_train)*100

    print("Training acuracy : ",TrainingAccuracy)
    print("Number of Training Iterations : ",model.n_iter_)

def GenerateConfusionMatrix(Y_test,Y_pred):
    print(Border)
    print("Generate Confusion Matrix ")
    print(Border)

    CM = confusion_matrix(Y_test,Y_pred)
    print("Confusion Matrix : ")
    print(CM)

def PlotLossCurve(model):
    print(Border)
    print("Plot the Loss Curve")
    print(Border)

    plt.plot(model.loss_curve_)
    plt.xlabel("iterarions")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.show()

def PredictAttrition(model,scaler):
    print(Border)
    print("Test the model with 5 new records")
    print(Border)

    NewEmployee = pd.DataFrame([
        [25, 3000, 2, 3, 5, 3, 3, 0, 1, 2],
        [45, 9000, 10, 15, 20, 2, 2, 1, 4, 3],
        [30, 4500, 4, 6, 8, 4, 3, 0, 2, 2],
        [50, 12000, 15, 25, 10, 2, 2, 1, 6, 1],
        [28, 3500, 3, 5, 4, 3, 4, 0, 1, 3]
    ],
    columns=['Age', 
             'MonthlyIncome', 
             'YearsAtCompany', 
             'TotalWorkingYears',
             'DistanceFromHome', 
             'JobSatisfaction', 
             'WorkLifeBalance', 
             'OverTime',
             'NumCompaniesWorked', 
             'TrainingTimesLastYear'
             ])

    NewEmployeeScaled = scaler.transform(NewEmployee)
    Prediction = model.predict(NewEmployeeScaled)

    for i in range(len(Prediction)):
        if Prediction[i] == 1:
            Result = "Employee May leave the company"
        else:
            Result = "Employee likely to stay"

        print("Employee",i+1,":",Result)

def main():
    df = LoadData()

    AnalyzeData(df)

    CheckMissing(df)

    DisplayFeatures(df)

    df = CategoricalToNumerical(df)

    print(df.head())

    X,Y = SeparateVariable(df)

    X_train,X_test,Y_train,Y_test = DivideDataset(df,X,Y)

    X_train,X_test, scaler = FeatureScaling(df,X_train,X_test)

    model = ModelCreation()

    model = ModelTraining(model,X_train,Y_train)

    Y_pred = ModelTesting(model,X_test,Y_test)

    ModelTrainingAccuracy(model,X_train,Y_train)

    GenerateConfusionMatrix(Y_test,Y_pred)

    PlotLossCurve(model)

    PredictAttrition(model,scaler)

if __name__ == "__main__":
    main()