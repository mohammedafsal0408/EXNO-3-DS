## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="1018" height="570" alt="image" src="https://github.com/user-attachments/assets/f5e28d8c-4f20-4a88-b805-e3e7ea002630" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="740" height="352" alt="image" src="https://github.com/user-attachments/assets/932fdcb7-2fa1-4b71-9838-95cb29a667a2" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="722" height="528" alt="image" src="https://github.com/user-attachments/assets/de7a0094-478a-4030-a5c3-1233bf5ab3cf" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="644" height="569" alt="image" src="https://github.com/user-attachments/assets/ae9e3176-7b2d-4b8c-a1b5-aeeb2c342bd0" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="550" height="441" alt="image" src="https://github.com/user-attachments/assets/d5d41fd8-1d19-4d95-83dc-e42d1511c695" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="828" height="456" alt="image" src="https://github.com/user-attachments/assets/4a8bb0a1-87d7-442c-a0cc-0a2de3803dd1" />

```
pip install --upgrade category_encoders
```
<img width="1382" height="428" alt="image" src="https://github.com/user-attachments/assets/7674c311-dc76-4e6d-87a9-7ec7891f369a" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="622" height="439" alt="image" src="https://github.com/user-attachments/assets/3c873135-9f9c-4812-89dd-03fdb150b654" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="620" height="444" alt="image" src="https://github.com/user-attachments/assets/917ab360-6361-4bfe-b2d7-67612a9bc129" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="877" height="441" alt="image" src="https://github.com/user-attachments/assets/0a8e403a-8e05-41c1-b849-14d05203bcff" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="709" height="445" alt="image" src="https://github.com/user-attachments/assets/b913b01e-9372-4af9-b175-75b1d94f7859" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="986" height="498" alt="image" src="https://github.com/user-attachments/assets/fd5db286-1738-4b93-8bc4-b649d9b973e1" />

```
df.skew()
```
<img width="434" height="243" alt="image" src="https://github.com/user-attachments/assets/d3b2306d-caaa-48b8-a21f-d0d2a3c34e84" />

```
np.log(df["Highly Positive Skew"])
```
<img width="470" height="557" alt="image" src="https://github.com/user-attachments/assets/fe502325-bf9b-4733-a6dc-14608fa651bd" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="536" height="586" alt="image" src="https://github.com/user-attachments/assets/e04f8fb2-6c47-46f2-8ff5-184219df79d4" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="597" height="577" alt="image" src="https://github.com/user-attachments/assets/da517d07-473c-4041-b896-baa0a7fb1f3b" />

```
np.square(df["Highly Positive Skew"])
```
<img width="651" height="567" alt="image" src="https://github.com/user-attachments/assets/e70d90b3-a010-4770-8d8a-db01808bd513" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1276" height="517" alt="image" src="https://github.com/user-attachments/assets/91a6a329-269d-4084-abb5-1683368027c4" />

```
df.skew()
```
<img width="515" height="294" alt="image" src="https://github.com/user-attachments/assets/48456f02-ee5e-4ede-b6cf-6c7b07e4782e" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="615" height="354" alt="image" src="https://github.com/user-attachments/assets/bcafb1a5-e438-4f3a-b9af-b565a06bd4c0" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1343" height="556" alt="image" src="https://github.com/user-attachments/assets/51145182-1808-4120-810b-db9c4fc95248" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="910" height="561" alt="image" src="https://github.com/user-attachments/assets/a332d6e5-9280-4499-9169-92cc9ed09c70" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="929" height="562" alt="image" src="https://github.com/user-attachments/assets/5e3d0168-a794-4256-a293-295b2d4d947a" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="1062" height="557" alt="image" src="https://github.com/user-attachments/assets/4633f19e-6c50-429c-aff0-1576bcb69c57" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="828" height="546" alt="image" src="https://github.com/user-attachments/assets/b23975b6-ed64-4509-9728-373717c2e5e7" />

```
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
<img width="1360" height="620" alt="image" src="https://github.com/user-attachments/assets/ee83fcdd-10f6-4738-857d-70e4d64cb454" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="1090" height="558" alt="image" src="https://github.com/user-attachments/assets/78c78d95-8d30-4a64-b102-f4e5d213fbc1" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="831" height="559" alt="image" src="https://github.com/user-attachments/assets/c40c22ff-2fed-4247-a40c-b043aa8b0944" />



# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
