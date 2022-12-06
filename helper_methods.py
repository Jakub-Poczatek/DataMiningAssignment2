import phik
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def analyse_discrete_numerical(dt, srs):
    # Print tables
    print_describe_and_info(srs)

    # Display plot
    fig, axs = plt.subplots()
    countPlot = sns.countplot(data=dt, x=srs, ax=axs)
    countPlot.bar_label(countPlot.containers[0])
    return fig

def analyse_continous_numerical(dt, srs):
    # Print tables
    print_describe_and_info(srs)

    # Combine and display plots
    fig, axs = plt.subplots(ncols=3)
    fig.set_figwidth(fig.get_figwidth() * 3)
    sns.boxplot(data=dt, x=srs, ax=axs[0])
    sns.ecdfplot(data=dt, x=srs, ax=axs[1])
    sns.kdeplot(data=dt, x=srs, ax=axs[2], multiple="stack")
    return fig

def correlate_discrete_continous(dataFrame, discFeature, contFeature):
    # Print tables
    matrix = dataFrame[[discFeature, contFeature]].phik_matrix()
    describe = dataFrame.groupby(discFeature)[contFeature].describe()
    print_side_by_side([matrix, describe])

    # Combine and display plots
    fig, axs = plt.subplots(ncols=3)
    fig.set_figwidth(fig.get_figwidth() * 3)
    sns.boxplot(data=dataFrame, x=contFeature, y=discFeature, orient="h", ax=axs[0])
    sns.barplot(data=dataFrame, x=discFeature, y=contFeature, ax=axs[1])
    sns.stripplot(data=dataFrame, x=discFeature, y=contFeature, ax=axs[2])
    return fig

def correlate_continous_numeric(dataFrame, baseFeature, compFeature):
    # Print tables
    pearson = dataFrame[[baseFeature, compFeature]].corr()
    spearman = dataFrame[[baseFeature, compFeature]].corr(method="spearman")
    print_side_by_side([pearson, spearman])

    # Combine and display plots
    fig, axs = plt.subplots(ncols=2)
    fig.set_figwidth(fig.get_figwidth() * 2)
    sns.regplot(data=dataFrame, x=baseFeature, y=compFeature, ax=axs[0], line_kws={"color": "red"})
    sns.lineplot(data=dataFrame, x=baseFeature, y=compFeature, ax=axs[1])
    return fig

def correlate_feature_target_discrete(dataFrame, feature, targets):
    # Print tables
    phik = dataFrame[[feature, targets[0], targets[1], targets[2]]].phik_matrix()
    describeCnt = dataFrame.groupby(feature)[targets[0]].describe()
    describeCasual = dataFrame.groupby(feature)[targets[1]].describe()
    describeRegistered = dataFrame.groupby(feature)[targets[2]].describe()
    print_side_by_side([phik, describeCnt])
    print_side_by_side([describeCasual, describeRegistered])

    # Combine and display plots
    fig, axs = plt.subplots(3, 3)
    fig.set_figwidth(fig.get_figwidth() * 3)
    fig.set_figheight(fig.get_figheight() * 3)
    sns.boxplot(data=dataFrame, x=targets[0], y=feature, orient="h", ax=axs[0, 0])
    sns.boxplot(data=dataFrame, x=targets[1], y=feature, orient="h", ax=axs[0, 1])
    sns.boxplot(data=dataFrame, x=targets[2], y=feature, orient="h", ax=axs[0, 2])

    sns.barplot(data=dataFrame, x=feature, y=targets[0], ax=axs[1, 0])
    sns.barplot(data=dataFrame, x=feature, y=targets[1], ax=axs[1, 1])
    sns.barplot(data=dataFrame, x=feature, y=targets[2], ax=axs[1, 2])

    sns.stripplot(data=dataFrame, x=feature, y=targets[0], ax=axs[2, 0])
    sns.stripplot(data=dataFrame, x=feature, y=targets[1], ax=axs[2, 1])
    sns.stripplot(data=dataFrame, x=feature, y=targets[2], ax=axs[2, 2])
    return fig

def correlate_feature_target_continous(dataFrame, feature, targets):
    # Print tables
    pearson = dataFrame[[feature, targets[0], targets[1], targets[2]]].corr()
    spearman = dataFrame[[feature, targets[0], targets[1], targets[2]]].corr(method="spearman")
    print_side_by_side([pearson, spearman])

    # Combine and display plots
    fig, axs = plt.subplots(2, 3)
    fig.set_figwidth(fig.get_figwidth() * 3)
    fig.set_figheight(fig.get_figheight() * 2)
    sns.regplot(data=dataFrame, x=feature, y=targets[0], ax=axs[0, 0], line_kws={"color": "red"})
    sns.regplot(data=dataFrame, x=feature, y=targets[1], ax=axs[0, 1], line_kws={"color": "red"})
    sns.regplot(data=dataFrame, x=feature, y=targets[2], ax=axs[0, 2], line_kws={"color": "red"})

    sns.lineplot(data=dataFrame, x=feature, y=targets[0], ax=axs[1, 0])
    sns.lineplot(data=dataFrame, x=feature, y=targets[1], ax=axs[1, 1])
    sns.lineplot(data=dataFrame, x=feature, y=targets[2], ax=axs[1, 2])
    return fig

def print_describe_and_info(srs):
    buffer = io.StringIO()
    srs.info(buf=buffer)

    # Format Describe
    dString = ("Describe: \n" +
               "====================\n" +
               srs.describe().to_string())

    # Format Info
    iString = ("\t\t\tInfo:\n" +
               "====================\n" +
               buffer.getvalue())

    # Display Describe and Info side by side
    for i, line in enumerate(dString.splitlines()):
        print(dString.splitlines()[i] + "\t\t" + iString.splitlines()[i])
        
    print("\n")

# Work in progress - needs improvement
def print_side_by_side(dataFrameArray):
    print("\n")

    for i, frame in enumerate(dataFrameArray):
        frame = frame.to_string().splitlines()
        dataFrameArray[i] = frame

    maxAmount = len(max(dataFrameArray, key=len))

    for i, frame in enumerate(dataFrameArray):
        if(len(frame) < maxAmount):
            multiplier = maxAmount - len(frame)
            lineLen = len(max(frame, key=len))
            for j in range(multiplier):
                frame.append((" "*lineLen)) # Alt-255
        dataFrameArray[i] = frame

    for i in range(maxAmount):
        print(dataFrameArray[0][i] + "\t\t" + dataFrameArray[1][i])
        
    print("\n")

def calculatePredictors(x, y):
    nP = x.shape[1]
    scores = [0]
    foundPredictors = list()

    for i in range(nP): # loop over all columns (predictors) in x
        (score, bestPredictorFound) = findNextBestPredictor(x, foundPredictors, y)
        foundPredictors.append(bestPredictorFound)
        scores.append(score)

    print(foundPredictors)
    print(scores)
    return scores

def calculatePolynomialScore(x, y):
    model = LinearRegression()
    scores = []
    for i in range(1, 13):
        poly = PolynomialFeatures(degree=i)
        xPoly = poly.fit_transform(x)
        model.fit(xPoly, y)
        scores.append(model.score(xPoly, y))
    return scores

def findNextBestPredictor(X, foundPredictors, y):
    model = linear_model.LinearRegression()
    nP = X.shape[1] # number of columns in X
    allPredictors = list(X) # See https://stackoverflow.com/a/19483025
    predictorsToSearch = set(allPredictors) - set(foundPredictors)
    maxScore = 0 # can usually do better than this!
    for predictor in predictorsToSearch: # loop over all remaining columns (predictors) in X
        trialPredictors = set(foundPredictors)
        trialPredictors.add(predictor) # Add this predictor to the existing predictors
        XcolSubset = X.loc[:,list(trialPredictors)] # all rows and just the trial predictors
        model.fit(XcolSubset, y) # fit the model to y
        score = model.score(XcolSubset, y)
        if score > maxScore: # identify the largest score and its associated predictor
            maxScore = score
            bestPredictorFound = predictor

    return (maxScore, bestPredictorFound)

def calculateDegrees(x, y, testX, predictors):
    scores = []
    newModel = LinearRegression()
    for i in range(1, 13):
        poly = PolynomialFeatures(degree=i)
        xPoly = poly.fit_transform(x)
        newModel.fit(xPoly, y)

        entry = testX.copy()
        polyEntry = entry[predictors]
        polyEntry = poly.fit_transform(polyEntry)
        predict = newModel.predict(polyEntry)
        actual = entry.casual
        scores.append(abs((actual - predict)).mean())
    return scores

