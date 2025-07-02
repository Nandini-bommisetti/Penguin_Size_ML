# Penguin_Size_ML
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
df = pd.read_csv('penguins_size.csv')
print(df)
#coverting all values to 0's and 1's
x = df['culmen_length_mm'].mean()
df['culmen_length_mm'] = df['culmen_length_mm'].fillna(x)
y = df['culmen_depth_mm'].mean()
df['culmen_depth_mm'] = df['culmen_depth_mm'].fillna(y)
z = df['flipper_length_mm'].mean()
df['flipper_length_mm'] = df['flipper_length_mm'].fillna(z)
a = df['body_mass_g'].mean()
df['body_mass_g'] = df['body_mass_g'].fillna(a)

print("The no.of zeroes in culmen_length_mm",df[df['culmen_length_mm']==0].shape[0])
df['culmen_length_mm'] = df['culmen_length_mm'].replace(0, df['culmen_length_mm'].mean()) 
print("The no.of zeroes in culmen_depth_mm",df[df['culmen_depth_mm']==0].shape[0])
df['culmen_depth_mm'] = df['culmen_depth_mm'].replace(0, df['culmen_depth_mm'].mean()) 
print("The no.of zeroes in flipper_length_mm",df[df['flipper_length_mm']==0].shape[0])
df['flipper_length_mm'] = df['flipper_length_mm'].replace(0, df['flipper_length_mm'].mean()) 
print("The no.of zeroes in body_mass_g",df[df['body_mass_g']==0].shape[0])
df['body_mass_g'] = df['body_mass_g'].replace(0, df['body_mass_g'].mean()) 
#labelencoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df['island'] = le.fit_transform(df['island'])
df['sex'] = le.fit_transform(df['sex'])
print(df)
#numbers to binary forms
ss = StandardScaler()
df['culmen_length_mm'] = ss.fit_transform(df[['culmen_length_mm']])
df['culmen_depth_mm'] = ss.fit_transform(df[['culmen_depth_mm']])
df['flipper_length_mm'] = ss.fit_transform(df[['flipper_length_mm']])
df['body_mass_g'] = ss.fit_transform(df[['body_mass_g']])
#minmaxscaler
ms = MinMaxScaler()
df['culmen_length_mm'] = ms.fit_transform(df[['culmen_length_mm']])
df['culmen_depth_mm'] = ms.fit_transform(df[['culmen_depth_mm']])
df['flipper_length_mm'] = ms.fit_transform(df[['flipper_length_mm']])
df['body_mass_g'] = ms.fit_transform(df[['body_mass_g']])
print(df)
# Save the processed DataFrame to a new CSV file
df.to_csv('updated_data.csv', index=False)
# train the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import r2_score
# Prepare the data for training
df=pd.read_csv('updated_data.csv')
x=df[['species','island','culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g']]
y=df['sex']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
result = r2_score(y_test,y_pred)
print(result)
a = int(input('enter species'))
b = int(input('enter island'))
c = float(input('enter culmen_length_mm'))
d = float(input('enter culmen_depth_mm'))
e = float(input('enter flipper_length_mm'))
f = float(input('enter body_mass_g'))
ans = model.predict([[a,b,c,d,e,f]])
print(ans)
