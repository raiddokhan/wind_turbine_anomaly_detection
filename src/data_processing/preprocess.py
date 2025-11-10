import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This section is for handling NaNs in the data

def preprocess_dataset(dataset: pd.DataFrame):

    ## Handle missing data -> data is complete, so no further process needed
    count_nans = len(dataset) - len(dataset.dropna())
    print("Number of NaNs in this dataset:", count_nans)

    ##### Feature selection #####
    # Delete the Unnamed: 0 column
    dataset = dataset.drop("Unnamed: 0", axis=1)
    print(dataset.columns.tolist())


    ##### Handle outliers #####
    print(dataset.info())
    # We handle the outliers by getting the value at which 1% and 99% of the data lie. 
    # Every value which is above the 99% and below the 1% will get the value for lower_quantile and upper_quantile 
    outliercheck_features = ["Temp_2m", "RelHum_2m", "DP_2m", "WS_10m", "WS_100m", "WG_10m", "Power"]
    for col in outliercheck_features:
        lower_quantile = dataset[col].quantile(0.01)
        upper_quantile = dataset[col].quantile(0.99)
        dataset[col] = dataset[col].clip(lower_quantile, upper_quantile)
    # Boxplot visualization of the features in outliercheck_features
    for col in outliercheck_features:
        plt.figure(figsize=(6,3))
        sns.boxplot(x=dataset[col])
        plt.title(col)
        plt.show()

    ##### Feature engineering #####
    # Since we have a WD in degrees, we have the issue that 359 and 1 degree are far apart 
    # We can use sin and cos -> This way we will have similar sin and cos results for 359 and 1 degree, which will improve the training
    # This means: For each WD column, we will replace it with two columns (sin and cos equivalents to represent the WD
    dataset["Sin_WD_10m"] = np.sin(dataset["WD_10m"])
    dataset["Cos_WD_10m"] = np.cos(dataset["WD_10m"])
    dataset["Sin_WD_100m"] = np.sin(dataset["WD_100m"])
    dataset["Cos_WD_100m"] = np.cos(dataset["WD_100m"])
    dataset = dataset.drop(columns = ["WD_10m", "WD_100m"])

    # Now we can also make use of sin and cos to transfor the Time Objects into sin cos features
    # Reason 1: days, months, hours and mins are all cyclical -> Hour 23 and Hour 1 are closer together as sin cos equivalents (like for wind directions)
    # Reason 2: Doing the same with one hot encoding would create date with high dimensions -> the approach with sin and cos does not
    dataset["Time"] = pd.to_datetime(dataset["Time"], format="%d-%m-%Y %H:%M")
    hour = dataset["Time"].dt.hour
    # min = dataset["Time"].dt.minute
    day = dataset["Time"].dt.day
    month = dataset["Time"].dt.month
    year = dataset["Time"].dt.year
    dataset["Sin_Hour"] = np.sin(2 * np.pi * hour/24)
    dataset["Cos_Hour"] = np.cos(2 * np.pi * hour/24)
    # dataset["Sin_min"] = np.sin(2 * np.pi * min/60)
    # dataset["Cos_min"] = np.cos(2 * np.pi * min/60)
    dataset["Sin_Day"] = np.sin(2 * np.pi * day / dataset["Time"].dt.days_in_month)
    dataset["Cos_Day"] = np.cos(2 * np.pi * day / dataset["Time"].dt.days_in_month)
    dataset["Sin_Month"] = np.sin(2 * np.pi * month / 12)
    dataset["Cos_Month"] = np.cos(2 * np.pi * month / 12)
    dataset["Year"] = year
    dataset = dataset.drop(columns=["Time"])
    # Uncomment before training
    # dataset = dataset.drop(columns=["Time"]) 
    # print(dataset[["Time","Sin_hour", "Cos_hour", "Sin_day", "Cos_day", "Sin_month", "Cos_month", "Year"]])

    # Location one hot encoding
    dataset = pd.get_dummies(dataset, columns=["Location"], prefix="Location")

    # WS100m - WS10m -> Reason: Indicates if there are turbulences on different levels
    dataset["Wind_Shear"] = dataset["WS_100m"] - dataset["WS_10m"]
    print(dataset["Wind_Shear"])

    # temp x relhum -> Felt temperature -> has influence on power generation ability
    dataset["Felt_Temp"] = dataset["Temp_2m"] * dataset["RelHum_2m"]
    

    ##### Normalize/Scale #####
    # Standardization for Features -> "Temp_2m","RelHum_2m","DP_2m","WS_10m","WS_100m","WG_10m","Wind_Shear","Felt_Temp","Year" 
    scalable_features = ["Temp_2m","RelHum_2m","DP_2m","WS_10m","WS_100m","WG_10m","Wind_Shear","Felt_Temp","Year"]
    dataset[scalable_features] = (dataset[scalable_features] - dataset[scalable_features].mean()) / dataset[scalable_features].std()

    ##### Build sequences #####

    



train_set = pd.read_csv(r"D:\Coding Projects\Machine Learning Projects\wind_turbine_anomaly_detection\data\Train.csv")
test_set = pd.read_csv(r"D:\Coding Projects\Machine Learning Projects\wind_turbine_anomaly_detection\data\Test.csv")

preprocess_dataset(train_set)