# Importing required libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Graph
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Credit Approval Prediction (DT)",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


st.subheader(':blue[Credit Approval Prediction by Decision Tree Classifier]')
st.write('Applicant ID , Credit Score, Income, Loan Amount, Loan Term , (Approved) ' )

if st.button('Predict', type='primary'):
    with st.spinner('Processing'):
        # Step 1: Sample Training Dataset Creation
        sample_size = 500

        data = {
            'Applicant ID': range(1, sample_size + 1),
            'Credit Score': [650, 700, 600, 720, 680, 750, 620, 700, 710, 690] * (sample_size // 10),
            'Income': [50000, 60000, 40000, 80000, 55000, 90000, 45000, 70000, 75000, 60000] * (sample_size // 10),
            'Loan Amount': [10000, 20000, 15000, 30000, 25000, 35000, 18000, 28000, 22000, 32000] * (sample_size // 10),
            'Loan Term': [12, 24, 12, 36, 24, 48, 12, 36, 24, 60] * (sample_size // 10),
            'Approved': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes'] * (sample_size // 10)
        }
        # Step 2: DataFrame Creation
        df = pd.DataFrame(data)
        col1 , col2 , col3 = st.columns(3)
        with col1:
            # Displaying data frames
            st.write(f"✅ 1. Generate training Data: have {str(len(df))} records")
            st.write(df.head(n=20))

        with col2:
            # Step 3: Data Preprocessing
            # Encode 'Approved' column with numerical labels
            st.write('✅ 2. Data cleaning - encoding target variable (Approved) Yes: 1, No: 0 -> df Y ')
            approval_mapping = {'Yes': 1, 'No': 0}
            # where 1 represents 'Yes' (approved) and 0 represents 'No' (not approved). 
            df['Approved'] = df['Approved'].map(approval_mapping)
            # Step 4: Train-Test Split
            st.write('✅  Create X  - Drop Applicant & Approved feature')
            X = df.drop(['Applicant ID', 'Approved'], axis=1)
            st.write('🟧  X DF')
            st.write(X.head(10))

        with col3:
            y = df['Approved']
            st.write('✅  3. Create Y df after extract target from source')
            st.write('🟦 Y  DF')
            st.write(y.head(10))


        st.write('♻️ Split Data (8 : 2) into X_train, X_test, y_train, y_test ')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 5: Model Training
        st.write('✅ 4. Create a decision tree model')
        clf = DecisionTreeClassifier()
        st.write('✅  Train the decision tree model by X_train & y_train')
        clf.fit(X_train, y_train)

        col1, col2 = st.columns(2)
        with col1:
            st.write('✅  ⬛️ X_train - 80% , all input variables')
            st.write(X_train.head(3))
            st.write('✅  ⬛️ X_test - 20% , all input variables')
            st.write(X_test.head(3))


        with col2:
            st.write('✅  ⬛️ y_train - 80% , 1 target column')
            st.write(y_train.head(3))
            st.write('✅  ⬛️  y_test - 20% , 1 target column')
            st.write(y_test.head(3))

                     
        col1, col2 = st.columns(2)
        with col1:
            st.write('Create test data - unseen by model , same format but without the target variable')
            # Step 6: Testing Data Creation
            testing_data = {
                'Applicant ID': range(11, 16),
                'Credit Score': [680, 720, 650, 700, 750],
                'Income': [55000, 80000, 50000, 60000, 90000],
                'Loan Amount': [18000, 30000, 10000, 20000, 35000],
                'Loan Term': [12, 36, 12, 24, 48],
                'Approved': ['No', 'Yes', 'No', 'Yes', 'Yes']
            }

            # Step 7: DataFrame Creation for Testing Data
            df_test = pd.DataFrame(testing_data)
            st.write("✅  5. 🔷 Testing Data: (before model train)")
            st.write(df_test.head(n=20))


            # Step 8: Data Preprocessing for Testing Data
            df_test['Approved'] = df_test['Approved'].map(approval_mapping)

        with col2:
            # Step 9: Model Prediction and Display
            # drop the  target column 
            st.write('✅  6. model prediction : make predictions by test data using the trained decision tree model. ')
            test_predictions = clf.predict(df_test.drop(['Applicant ID', 'Approved'], axis=1))
            st.write('✅  model prediction : create an array of predicted labels ')
            st.write(test_predictions)
        
        col1 , col2 = st.columns(2)
        with col1:
            df_test['Predicted Approval'] = test_predictions
            st.write('✅  7. 🔲  model prediction : add [Predicted Approval] column  to  test data')
            st.write(df_test.head(n=20))

        with col2:
            plt.figure(figsize=(10, 6))
            plt.scatter(df_test['Income'], df_test['Approved'], label='Approval', marker='o', color='b', alpha=0.6)
            plt.scatter(df_test['Income'], df_test['Predicted Approval'], label='Predict Approval', marker='x', color='r', alpha=0.9)
            plt.xlabel('Approval')
            plt.ylabel( 'Income $')
            plt.title('<Approval> correlated with Income $ ')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

        col1 , col2 = st.columns(2)
        with col1:
            # Group the data by 'Income' and calculate the count of 'Yes' and 'No' for each income level
            income_approval_counts = df_test.groupby(['Income', 'Predicted Approval']).size().unstack()
            # Plot the bar chart
            income_approval_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.xlabel('Income Level')
            plt.ylabel('Count')
            plt.title('Credit Approval by Income Level')
            plt.legend(title='Credit Approval', loc='upper right', labels=['No', 'Yes'])
            st.pyplot(plt)

        with col2:
            # Group the data by 'Income' and calculate the count of 'Yes' and 'No' for each income level
            income_approval_counts = df_test.groupby(['Credit Score', 'Predicted Approval']).size().unstack()
            # Plot the bar chart
            income_approval_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.xlabel('Income Level')
            plt.ylabel('Count')
            plt.title('Credit Approval by Income Level')
            plt.legend(title='Credit Approval', loc='upper right', labels=['No', 'Yes'])
            st.pyplot(plt)


 

st.write('CustomerChurn-DTree.py.py')