import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

df = pd.read_csv('tech_test_data.csv')
df_clean = df.dropna().reset_index(drop=True)

le = LabelEncoder()
df_clean['behavior_encoded'] = le.fit_transform(df_clean['behaviour'])

columns_to_exclude = ['behaviour', 'cow_id', 'observation_id', 'behavior_encoded']
features = df_clean.drop(columns=columns_to_exclude)
target = df_clean['behavior_encoded']

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

target_names = le.classes_
report_dict = classification_report(y_test, predictions, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
pd.set_option('float_format', '{:.3f}'.format)
report_df_formatted = report_df

print(f'Accuracy: {accuracy:.3f}')
print('Classification Report:')
print(report_df_formatted)

cv_scores = cross_val_score(xgb_model, features_scaled, target, cv=5, scoring='accuracy')
print(f'Cross-validated Accuracy: {cv_scores.mean():.3f}')