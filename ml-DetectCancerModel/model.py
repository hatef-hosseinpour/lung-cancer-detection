import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, r2_score
import numpy as np
import pickle


class Model:
    def __init__(self, dataFile="survey lung cancer.csv"):

        self.df = pd.read_csv(dataFile)

        pd.set_option("display.max.rows", None)
        pd.set_option("display.max.columns", None)

        self.dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
        self.rnd = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
        self.knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

        self.voting = VotingClassifier(
            estimators=[
                ("decision_tree", self.dt),
                ("random_forest", self.rnd),
                ("k_neighbors", self.knn),
            ],
            voting="hard"
        )
   #################### Split df into to train set and test set ############################################

    def splitDf(self, testSize):
        # print(len(self.df))

        # Remove all duplicate rows
        self.df.drop_duplicates(inplace=True)

        # print(len(self.df))

        # after feature selection this feature will be removed
        self.df.drop(
            self.df.columns[[3, 7, 8, 9, 10, 13]], axis=1, inplace=True)

        self.x = self.df.drop(columns=["LUNG_CANCER"])  # independent value
        self.y = self.df["LUNG_CANCER"]  # dependent value Class Lable

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=testSize, random_state=10, shuffle=True)

        # print(self.df.isnull().sum())

    ################################## Feature Scaling #####################################################

    def fit(self):

        self.sc = StandardScaler()
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

        model = self.voting.fit(self.x_train, self.y_train)
        
        return model

    def predict(self):

        y_pred = self.voting.predict(self.x_test)

        Accuracy = accuracy_score(self.y_test, y_pred)
        Conf_Matrix = confusion_matrix(self.y_test, y_pred)

        # tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        # Accuracy = ((tp+tn)/(tn+fp+fn+tp))
        
        return [Accuracy, Conf_Matrix]

    def featureSelection(self):

        bestfeatures = SelectKBest(score_func=chi2, k=9)
        fit = bestfeatures.fit(self.x, self.y)

        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(self.x.columns)

        # concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        # naming the dataframe columns
        featureScores.columns = ["Specs", "Score"]
        selectedFeatures = featureScores.nlargest(9, "Score")

        return selectedFeatures
        # return self.df.describe()


if __name__ == "__main__":
    model_instance = Model()

    model_instance.splitDf(0.2)

    model = model_instance.fit()

    # pickle.dump(model, open('model.pkl', 'wb'))

    pred = model_instance.predict()

    features = model_instance.featureSelection()
    # print(features)

    print(f"\n \n ACCURACY = {pred[0]} \n \n CONFUSION_MATRIX : \n \n {pred[1]} \n ")
