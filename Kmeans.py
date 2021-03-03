import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
airlines = pd.read_excel("C:/Users/hp/Desktop/kmeans_assi/EastWestAirlines.xlsx" , sheet_name="data")
airlines.drop(['ID#'] , axis=1 , inplace=True)
airlines.columns

airlines.isna().sum()
airlines.isnull().sum()

dups = airlines.duplicated()
sum(dups)

airlines = airlines.drop_duplicates()

seaborn.boxplot(airlines.Balance);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.Qual_miles);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.cc1_miles);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.cc2_miles);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.cc3_miles);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.Bonus_miles);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.Bonus_trans);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.Flight_miles_12mo);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.Flight_trans_12);plt.title("Boxplot");plt.show()
seaborn.boxplot(airlines.Days_since_enroll);plt.title("Boxplot");plt.show()

plt.scatter(airlines["Balance"] , airlines["Bonus_miles"])
plt.scatter(airlines["Balance"] , airlines["Bonus_trans"])
plt.scatter(airlines["Flight_miles_12mo"] , airlines["Flight_trans_12"])
plt.scatter(airlines["Bonus_miles"] , airlines["Bonus_trans"])

IQR = airlines["Balance"].quantile(0.75) - airlines["Balance"].quantile(0.25)
L_limit_balance = airlines["Balance"].quantile(0.25) - (IQR * 1.5)
H_limit_balance = airlines["Balance"].quantile(0.75) + (IQR * 1.5)
airlines["Balance"] = pd.DataFrame(np.where(airlines["Balance"] > H_limit_balance , H_limit_balance ,
                                    np.where(airlines["Balance"] < L_limit_balance , L_limit_balance , airlines["Balance"])))
seaborn.boxplot(airlines.Balance);plt.title('Boxplot');plt.show()

IQR = airlines["Bonus_miles"].quantile(0.75) - airlines["Bonus_miles"].quantile(0.25)
L_limit_Bonus_miles = airlines["Bonus_miles"].quantile(0.25) - (IQR * 1.5)
H_limit_Bonus_miles = airlines["Bonus_miles"].quantile(0.75) + (IQR * 1.5)
airlines["Bonus_miles"] = pd.DataFrame(np.where(airlines["Bonus_miles"] > H_limit_Bonus_miles , H_limit_Bonus_miles ,
                                    np.where(airlines["Bonus_miles"] < L_limit_Bonus_miles , L_limit_Bonus_miles , airlines["Bonus_miles"])))
seaborn.boxplot(airlines.Bonus_miles);plt.title('Boxplot');plt.show()

IQR = airlines["Bonus_trans"].quantile(0.75) - airlines["Bonus_trans"].quantile(0.25)
L_limit_Bonus_trans = airlines["Bonus_trans"].quantile(0.25) - (IQR * 1.5)
H_limit_Bonus_trans = airlines["Bonus_trans"].quantile(0.75) + (IQR * 1.5)
airlines["Bonus_trans"] = pd.DataFrame(np.where(airlines["Bonus_trans"] > H_limit_Bonus_trans , H_limit_Bonus_trans ,
                                    np.where(airlines["Bonus_trans"] < L_limit_Bonus_trans , L_limit_Bonus_trans , airlines["Bonus_trans"])))
seaborn.boxplot(airlines.Bonus_trans);plt.title('Boxplot');plt.show()

IQR = airlines["Flight_miles_12mo"].quantile(0.75) - airlines["Flight_miles_12mo"].quantile(0.25)
L_limit_Flight_miles_12mo = airlines["Flight_miles_12mo"].quantile(0.25) - (IQR * 1.5)
H_limit_Flight_miles_12mo = airlines["Flight_miles_12mo"].quantile(0.75) + (IQR * 1.5)
airlines["Flight_miles_12mo"] = pd.DataFrame(np.where(airlines["Flight_miles_12mo"] > H_limit_Flight_miles_12mo , H_limit_Flight_miles_12mo ,
                                    np.where(airlines["Flight_miles_12mo"] < L_limit_Flight_miles_12mo , L_limit_Flight_miles_12mo , airlines["Flight_miles_12mo"])))
seaborn.boxplot(airlines.Flight_miles_12mo);plt.title('Boxplot');plt.show()

IQR = airlines["Flight_trans_12"].quantile(0.75) - airlines["Flight_trans_12"].quantile(0.25)
L_limit_Flight_trans_12 = airlines["Flight_trans_12"].quantile(0.25) - (IQR * 1.5)
H_limit_Flight_trans_12 = airlines["Flight_trans_12"].quantile(0.75) + (IQR * 1.5)
airlines["Flight_trans_12"] = pd.DataFrame(np.where(airlines["Flight_trans_12"] > H_limit_Flight_trans_12 , H_limit_Flight_trans_12 ,
                                    np.where(airlines["Flight_trans_12"] < L_limit_Flight_trans_12 , L_limit_Flight_trans_12 , airlines["Flight_trans_12"])))
seaborn.boxplot(airlines.Flight_trans_12);plt.title('Boxplot');plt.show()

IQR = airlines["Days_since_enroll"].quantile(0.75) - airlines["Days_since_enroll"].quantile(0.25)
L_limit_Days_since_enroll = airlines["Days_since_enroll"].quantile(0.25) - (IQR * 1.5)
H_limit_Days_since_enroll = airlines["Days_since_enroll"].quantile(0.75) + (IQR * 1.5)
airlines["Days_since_enroll"] = pd.DataFrame(np.where(airlines["Days_since_enroll"] > H_limit_Days_since_enroll , H_limit_Days_since_enroll ,
                                    np.where(airlines["Days_since_enroll"] < L_limit_Days_since_enroll , L_limit_Days_since_enroll , airlines["Days_since_enroll"])))
seaborn.boxplot(airlines.Days_since_enroll);plt.title('Boxplot');plt.show()


def norm_fun(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

airlines_norm = norm_fun(airlines.iloc[:,0:])

from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model_airlines = KMeans(n_clusters = 3)
model_airlines.fit(airlines_norm)

model_airlines.labels_ # getting the labels of clusters assigned to each row 
cluster_airlines = pd.Series(model_airlines.labels_)  # converting numpy array into pandas series object 
airlines['cluster'] = cluster_airlines

airlines = airlines.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
airlines.head()

airlines.iloc[:, 1:11].groupby(airlines.cluster).mean()

airlines.to_csv("Kmeans_airlines.csv", encoding = "utf-8")

import os
os.getcwd()

#################################Problem 2############################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import  numpy as np

crime_data = pd.read_csv("C:/Users/hp/Desktop/kmeans_assi/crime_data.csv")
crime_data.columns

crime_data.isna().sum()
crime_data.isnull().sum()

dups1 = crime_data.duplicated()
sum(dups1)

crime_data = crime_data.drop_duplicates()

seaborn.boxplot(crime_data.Murder);plt.title("Murder");plt.show()
seaborn.boxplot(crime_data.Assault);plt.title("Assault");plt.show()
seaborn.boxplot(crime_data.UrbanPop);plt.title("UrbanPop");plt.show()
seaborn.boxplot(crime_data.Rape);plt.title("Rape");plt.show()

plt.scatter(crime_data["Murder"] , crime_data["Assault"])
plt.scatter(crime_data["UrbanPop"] , crime_data["Rape"])
plt.scatter(crime_data["Murder"] , crime_data["Rape"])

IQR = crime_data["Rape"].quantile(0.75) - crime_data["Rape"].quantile(0.25)
L_limit_Rape = crime_data["Rape"].quantile(0.25) - (IQR * 1.5)
H_limit_Rape = crime_data["Rape"].quantile(0.75) + (IQR * 1.5)
crime_data["Rape"] = pd.DataFrame(np.where(crime_data["Rape"] > H_limit_Rape , H_limit_Rape ,
                                    np.where(crime_data["Rape"] < L_limit_Rape , L_limit_Rape , crime_data["Rape"])))
seaborn.boxplot(crime_data.Rape);plt.title('Boxplot');plt.show()

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

crime_data_norm = norm_func(crime_data.iloc[: , 1:])

from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model_crime = KMeans(n_clusters = 3)
model_crime.fit(crime_data_norm)

model_crime.labels_ # getting the labels of clusters assigned to each row 
cluster_crime = pd.Series(model_crime.labels_)  # converting numpy array into pandas series object 
crime_data['cluster'] = cluster_crime

crime_data = crime_data.iloc[:,[5,0,1,2,3,4]]
crime_data.head()

crime_data.iloc[:, 1:].groupby(crime_data.cluster).mean()

crime_data.to_csv("Kmeans_crime.csv", encoding = "utf-8")

import os
os.getcwd()
##################################Problem 3#############################################

import pandas as pd
import seaborn
import numpy as np
import matplotlib.pyplot as plt

insurance_data = pd.read_csv("C:/Users/hp/Desktop/kmeans_assi/Insurance Dataset.csv")

duplics = insurance_data.duplicated()
sum(duplics)

insurance_data.isna().sum()
insurance_data.columns

seaborn.boxplot(insurance_data["Premiums Paid"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Age"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Days to Renew"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Claims made"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Income"]);plt.title("Boxplot");plt.show()


plt.scatter(insurance_data["Premiums Paid"] , insurance_data["Age"])
plt.scatter(insurance_data["Days to Renew"] , insurance_data["Claims made"])
plt.scatter(insurance_data["Income"] , insurance_data["Premiums Paid"])

IQR = insurance_data["Premiums Paid"].quantile(0.75) - insurance_data["Premiums Paid"].quantile(0.25)
L_limit_Premiums_Paid = insurance_data["Premiums Paid"].quantile(0.25) - (IQR * 1.5)
H_limit_Premiums_Paid = insurance_data["Premiums Paid"].quantile(0.75) + (IQR * 1.5)
insurance_data["Premiums Paid"] = pd.DataFrame(np.where(insurance_data["Premiums Paid"] > H_limit_Premiums_Paid , H_limit_Premiums_Paid ,
                                    np.where(insurance_data["Premiums Paid"] < L_limit_Premiums_Paid , L_limit_Premiums_Paid , insurance_data["Premiums Paid"])))
seaborn.boxplot(insurance_data["Premiums Paid"]);plt.title('Boxplot');plt.show()

IQR = insurance_data["Claims made"].quantile(0.75) - insurance_data["Claims made"].quantile(0.25)
L_limit_Claims_made = insurance_data["Claims made"].quantile(0.25) - (IQR * 1.5)
H_limit_Claims_made = insurance_data["Claims made"].quantile(0.75) + (IQR * 1.5)
insurance_data["Claims made"] = pd.DataFrame(np.where(insurance_data["Claims made"] > H_limit_Claims_made , H_limit_Claims_made ,
                                    np.where(insurance_data["Claims made"] < L_limit_Claims_made , L_limit_Claims_made , insurance_data["Claims made"])))
seaborn.boxplot(insurance_data["Claims made"]);plt.title('Boxplot');plt.show()

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

insurance_norm = norm_func(insurance_data)

from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(insurance_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model_insurance = KMeans(n_clusters = 3)
model_insurance.fit(insurance_norm)

model_insurance.labels_ # getting the labels of clusters assigned to each row 
cluster_insurance = pd.Series(model_insurance.labels_)  # converting numpy array into pandas series object 
insurance_data['cluster'] = cluster_insurance

insurance_data = insurance_data.iloc[:,[5,0,1,2,3,4]]
insurance_data.head()

insurance_data.iloc[:, :].groupby(insurance_data.cluster).mean()

insurance_data.to_csv("Kmeans_insurance.csv", encoding = "utf-8")

import os
os.getcwd()
###########################################Program 4############################################


import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np

telco_data = pd.read_excel("C:/Users/hp/Desktop/kmeans_assi/Telco_customer_churn.xlsx")
telco_data.drop(['Count' , 'Quarter'] , axis=1 , inplace=True)

telco_data.isna().sum()

dupis = telco_data.duplicated()
sum(dupis)

telco_data = telco_data.drop_duplicates()

new_telco_data = pd.get_dummies(telco_data)

from sklearn.preprocessing import  OneHotEncoder

OH_enc = OneHotEncoder()

new_telco_data2 = pd.DataFrame(OH_enc.fit_transform(telco_data).toarray())

from sklearn.preprocessing import  LabelEncoder
L_enc = LabelEncoder()
telco_data['Referred a Friend'] = L_enc.fit_transform(telco_data['Referred a Friend'])
telco_data['Offer'] = L_enc.fit_transform(telco_data['Offer'])
telco_data['Phone Service'] = L_enc.fit_transform(telco_data['Phone Service'])
telco_data['Multiple Lines'] = L_enc.fit_transform(telco_data['Multiple Lines'])
telco_data['Internet Service'] = L_enc.fit_transform(telco_data['Internet Service'])
telco_data['Internet Type'] = L_enc.fit_transform(telco_data['Internet Type'])
telco_data['Online Security'] = L_enc.fit_transform(telco_data['Online Security'])
telco_data['Online Backup'] = L_enc.fit_transform(telco_data['Online Backup'])
telco_data['Device Protection Plan'] = L_enc.fit_transform(telco_data['Device Protection Plan'])
telco_data['Premium Tech Support'] = L_enc.fit_transform(telco_data['Premium Tech Support'])
telco_data['Streaming TV'] = L_enc.fit_transform(telco_data['Streaming TV'])
telco_data['Streaming Movies'] = L_enc.fit_transform(telco_data['Streaming Movies'])
telco_data['Streaming Music'] = L_enc.fit_transform(telco_data['Streaming Music'])
telco_data['Unlimited Data'] = L_enc.fit_transform(telco_data['Unlimited Data'])
telco_data['Contract'] = L_enc.fit_transform(telco_data['Contract'])
telco_data['Paperless Billing'] = L_enc.fit_transform(telco_data['Paperless Billing'])
telco_data['Payment Method'] = L_enc.fit_transform(telco_data['Payment Method'])


seaborn.boxplot(telco_data["Tenure in Months"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Avg Monthly Long Distance Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Avg Monthly GB Download"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Monthly Charge"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Total Refunds"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Extra Data Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Long Distance Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Revenue"]);plt.title("Boxplot");plt.show()

plt.scatter(telco_data["Tenure in Months"] , telco_data["Total Extra Data Charges"])
plt.scatter(telco_data["Monthly Charge"] , telco_data["Avg Monthly Long Distance Charges"])
plt.scatter(telco_data["Total Long Distance Charges"] , telco_data["Total Revenue"])

IQR = telco_data["Avg Monthly GB Download"].quantile(0.75) - telco_data["Avg Monthly GB Download"].quantile(0.25)
L_limit_Avg_Monthly_GB_Download = telco_data["Avg Monthly GB Download"].quantile(0.25) - (IQR * 1.5)
H_limit_Avg_Monthly_GB_Download = telco_data["Avg Monthly GB Download"].quantile(0.75) + (IQR * 1.5)
telco_data["Avg Monthly GB Download"] = pd.DataFrame(np.where(telco_data["Avg Monthly GB Download"] > H_limit_Avg_Monthly_GB_Download , H_limit_Avg_Monthly_GB_Download ,
                                    np.where(telco_data["Avg Monthly GB Download"] < L_limit_Avg_Monthly_GB_Download , L_limit_Avg_Monthly_GB_Download , telco_data["Avg Monthly GB Download"])))
seaborn.boxplot(telco_data["Avg Monthly GB Download"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Refunds"].quantile(0.75) - telco_data["Total Refunds"].quantile(0.25)
L_limit_Total_Refunds = telco_data["Total Refunds"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Refunds = telco_data["Total Refunds"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Refunds"] = pd.DataFrame(np.where(telco_data["Total Refunds"] > H_limit_Total_Refunds , H_limit_Total_Refunds ,
                                    np.where(telco_data["Total Refunds"] < L_limit_Total_Refunds , L_limit_Total_Refunds , telco_data["Total Refunds"])))
seaborn.boxplot(telco_data["Total Refunds"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Extra Data Charges"].quantile(0.75) - telco_data["Total Extra Data Charges"].quantile(0.25)
L_limit_Total_Extra_Data_Charges = telco_data["Total Extra Data Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Extra_Data_Charges = telco_data["Total Extra Data Charges"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Extra Data Charges"] = pd.DataFrame(np.where(telco_data["Total Extra Data Charges"] > H_limit_Total_Extra_Data_Charges , H_limit_Total_Extra_Data_Charges ,
                                    np.where(telco_data["Total Extra Data Charges"] < L_limit_Total_Extra_Data_Charges , L_limit_Total_Extra_Data_Charges , telco_data["Total Extra Data Charges"])))
seaborn.boxplot(telco_data["Total Extra Data Charges"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Long Distance Charges"].quantile(0.75) - telco_data["Total Long Distance Charges"].quantile(0.25)
L_limit_Total_Long_Distance_Charges = telco_data["Total Long Distance Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Long_Distance_Charges = telco_data["Total Long Distance Charges"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Long Distance Charges"] = pd.DataFrame(np.where(telco_data["Total Long Distance Charges"] > H_limit_Total_Long_Distance_Charges , H_limit_Total_Long_Distance_Charges ,
                                    np.where(telco_data["Total Long Distance Charges"] < L_limit_Total_Long_Distance_Charges , L_limit_Total_Long_Distance_Charges , telco_data["Total Long Distance Charges"])))
seaborn.boxplot(telco_data["Total Long Distance Charges"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Revenue"].quantile(0.75) - telco_data["Total Revenue"].quantile(0.25)
L_limit_Total_Revenue = telco_data["Total Revenue"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Revenue = telco_data["Total Revenue"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Revenue"] = pd.DataFrame(np.where(telco_data["Total Revenue"] > H_limit_Total_Revenue , H_limit_Total_Revenue ,
                                    np.where(telco_data["Total Revenue"] < L_limit_Total_Revenue , L_limit_Total_Revenue , telco_data["Total Revenue"])))
seaborn.boxplot(telco_data["Total Revenue"]);plt.title('Boxplot');plt.show()

def std_fun(i):
    x = (i-i.mean()) / (i.std())
    return (x)

telco_data_norm = std_fun(new_telco_data)

str(telco_data_norm)

from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(telco_data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model_telco = KMeans(n_clusters = 3)
model_telco.fit(telco_data_norm)

model_telco.labels_ # getting the labels of clusters assigned to each row 
cluster_telco = pd.Series(model_telco.labels_)  # converting numpy array into pandas series object 
telco_data['cluster'] = cluster_telco

telco_data.head()

telco_data.iloc[:,:].groupby(telco_data.cluster).mean()

telco_data.to_csv("Kmeans_telco.csv", encoding = "utf-8")

import os
os.getcwd()


###############################Problem 5#############################################

import pandas as pd
import seaborn
import numpy as np
import matplotlib.pyplot as plt

auto_data = pd.read_csv("C:/Users/hp/Desktop/kmeans_assi/AutoInsurance.csv")

auto_data.drop(['Customer'] , axis= 1 , inplace = True)

new_auto_data = auto_data.iloc[ : ,1:]

duplis = new_auto_data.duplicated()
sum(duplis)

new_auto_data = new_auto_data.drop_duplicates()

new_auto_data.isna().sum()


dummy_auto_data = pd.get_dummies(new_auto_data)

seaborn.boxplot(new_auto_data["Customer Lifetime Value"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Income"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Monthly Premium Auto"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Months Since Last Claim"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(new_auto_data["Months Since Policy Inception"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Total Claim Amount"]);plt.title("Boxplot");plt.show()

plt.scatter(new_auto_data["Customer Lifetime Value"] , new_auto_data["Income"])
plt.scatter(new_auto_data["Monthly Premium Autos"] , new_auto_data["Months Since Last Claime"])
plt.scatter(new_auto_data["Months Since Policy Inception"] , new_auto_data["Total Claim Amount"])

IQR = new_auto_data["Customer Lifetime Value"].quantile(0.75) - new_auto_data["Customer Lifetime Value"].quantile(0.25)
L_limit_Customer_Lifetime_Value = new_auto_data["Customer Lifetime Value"].quantile(0.25) - (IQR * 1.5)
H_limit_Customer_Lifetime_Value = new_auto_data["Customer Lifetime Value"].quantile(0.75) + (IQR * 1.5)
new_auto_data["Customer Lifetime Value"] = pd.DataFrame(np.where(new_auto_data["Customer Lifetime Value"] > H_limit_Customer_Lifetime_Value , H_limit_Customer_Lifetime_Value ,
                                    np.where(new_auto_data["Customer Lifetime Value"] < L_limit_Customer_Lifetime_Value , L_limit_Customer_Lifetime_Value , new_auto_data["Customer Lifetime Value"])))
seaborn.boxplot(new_auto_data["Customer Lifetime Value"]);plt.title('Boxplot');plt.show()

IQR = new_auto_data["Monthly Premium Auto"].quantile(0.75) - new_auto_data["Monthly Premium Auto"].quantile(0.25)
L_limit_Monthly_Premium_Auto = new_auto_data["Monthly Premium Auto"].quantile(0.25) - (IQR * 1.5)
H_limit_Monthly_Premium_Auto = new_auto_data["Monthly Premium Auto"].quantile(0.75) + (IQR * 1.5)
new_auto_data["Monthly Premium Auto"] = pd.DataFrame(np.where(new_auto_data["Monthly Premium Auto"] > H_limit_Monthly_Premium_Auto , H_limit_Monthly_Premium_Auto ,
                                    np.where(new_auto_data["Monthly Premium Auto"] < L_limit_Monthly_Premium_Auto , L_limit_Monthly_Premium_Auto , new_auto_data["Monthly Premium Auto"])))
seaborn.boxplot(new_auto_data["Monthly Premium Auto"]);plt.title('Boxplot');plt.show()

IQR = new_auto_data["Total Claim Amount"].quantile(0.75) - new_auto_data["Total Claim Amount"].quantile(0.25)
L_limit_Total_Claim_Amount = new_auto_data["Total Claim Amount"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Claim_Amount = new_auto_data["Total Claim Amount"].quantile(0.75) + (IQR * 1.5)
new_auto_data["Total Claim Amount"] = pd.DataFrame(np.where(new_auto_data["Total Claim Amount"] > H_limit_Total_Claim_Amount , H_limit_Total_Claim_Amount ,
                                    np.where(new_auto_data["Total Claim Amount"] < L_limit_Total_Claim_Amount , L_limit_Total_Claim_Amount , new_auto_data["Total Claim Amount"])))
seaborn.boxplot(new_auto_data["Total Claim Amount"]);plt.title('Boxplot');plt.show()



def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

auto_data_norm = norm_func(dummy_auto_data)

from sklearn.cluster import KMeans

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(auto_data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model_auto = KMeans(n_clusters = 3)
model_auto.fit(auto_data_norm)

model_auto.labels_ # getting the labels of clusters assigned to each row 
cluster_auto = pd.Series(model_auto.labels_)  # converting numpy array into pandas series object 
auto_data['cluster'] = cluster_auto

auto_data = auto_data.iloc[:,[23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
auto_data.head()

auto_data.iloc[:, :].groupby(auto_data.cluster).mean()

auto_data.to_csv("Kmeans_auto.csv", encoding = "utf-8")

import os
os.getcwd()



#################################################END############################################