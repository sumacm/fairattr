import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
#label 0: Employee will stay
#label 1: Employee will leave
 'label_maps': [{'No': 'Employee will stay', 'Yes': 'Employee will leave'}],
  'protected_attribute_maps': [{0.0: 'Old', 1.0: 'Young'},
                               {0.0: 'Male', 1.0: 'Female'}]
}

class EmployeeDataset(StandardDataset):
    def __init__(self, label_name='Attrition',  favorable_classes=['No'],
                 protected_attribute_names=['Age','Gender'],
                 privileged_classes = [lambda x: x < 25,['Male']],
                 na_values=["unknown"],
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'employee', 'emp_attrition.csv')
        try:

            df = pd.read_csv(filepath, sep=',', na_values=na_values)
            features_to_keep = ['Age', 'Attrition', 'MaritalStatus', 'Gender',
                                'EducationField','DailyRate','DistanceFromHome','Education','TotalWorkingYears'
                                ,'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','BusinessTravel',
                                'MonthlyIncome','StockOptionLevel','PerformanceRating']
            categorical_features = ['MaritalStatus','EducationField','BusinessTravel']
            """
            df = pd.get_dummies(data=sorted(df), columns=categorical_features)
            
            categorical_features = ['EducationField','Gender','MaritalStatus']
            # generate binary values using Sklearn
            from sklearn.preprocessing import OneHotEncoder
            # creating instance of one-hot-encoder
            One_enc = OneHotEncoder(handle_unknown='ignore')
            # Passing gender variable
            enc_df = pd.DataFrame(One_enc.fit_transform(df[categorical_features]).toarray())
            # merge with original dataset on key values
            df = df.join(enc_df)
           
            df = pd.get_dummies(data=sorted(df), columns=categorical_features)
            features_to_keep = ['Age','DailyRate','DistanceFromHome','Education','TotalWorkingYears','EnvironmentSatisfaction','Attrition','MaritalStatus','Gender',
                                'HourlyRate','JobInvolvement','JobLevel']"""
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\t https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/home")
            print("\nunzip it and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'employee'))))
            import sys
            sys.exit(1)

        super(EmployeeDataset, self).__init__(df=df,label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            features_to_keep=features_to_keep,
            categorical_features=categorical_features,
            metadata=metadata,
            na_values=na_values)