[heart-attack-complications-classification](https://github.com/mertmetin1/heart-attack-complications-classification-/files/15411140/DataScienceProject.pdf)

**Project Report: Feature Selection and Classification in MATLAB**

**1. Introduction**

In this project, we aimed to perform feature selection and classification using MATLAB on a dataset containing accelerometer data. The dataset was obtained from an Excel file and consisted of features related to physical activity. Our objective was to select the most relevant features and build robust classification models to predict the activity class labels.

**2. Data Preprocessing**

We started by reading the Excel file containing the dataset into MATLAB. We removed the 'Sutun1' column as it was affecting the classification and stored the class labels in a separate variable. To handle missing values, we filled them with the mean of the respective columns. Next, we performed min-max normalization to scale the features between 0 and 1. Finally, we converted the data into a numeric matrix for further processing.
![image](https://github.com/mertmetin1/heart-attack-complications-classification-/assets/98667673/85dd44b1-de2f-4578-b2b4-89529556ace0)

**3. Feature Selection using ReliefF**

We applied ReliefF feature selection technique to select the top 10 most relevant features from the dataset. ReliefF evaluates the importance of features by considering their relevance and redundancy. The selected features were then used for subsequent analysis.

**4. Principal Component Analysis (PCA)**

We applied PCA to reduce the dimensionality of the dataset and visualize the relationships between the selected features. PCA helps in capturing the maximum variance in the data by transforming the features into a new set of orthogonal variables called principal components. We analyzed the eigenvectors, scores, and explained variance of the principal components obtained from PCA.

**5. Model Training and Evaluation**

We split the dataset into training and test sets using cvpartition for cross-validation. We trained and evaluated several classification models including Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), Decision Tree (DT), and Random Forest (RF). We measured the accuracy of each model and visualized their performance using confusion matrices.

**6. Results and Conclusion**

Our results showed that Random Forest achieved the highest accuracy among all the models tested, with an accuracy of 89%. Random Forest demonstrated robust performance in classifying activity labels based on the selected features. The project highlights the importance of feature selection and classification techniques in analyzing accelerometer data for activity recognition.
![image](https://github.com/mertmetin1/heart-attack-complications-classification-/assets/98667673/5f65d422-21fb-4d11-afd9-069315a5a621)

![image](https://github.com/mertmetin1/heart-attack-complications-classification-/assets/98667673/b1a4023a-3e63-4ff0-a5f2-f803fea96edc)

![image](https://github.com/mertmetin1/heart-attack-complications-classification-/assets/98667673/02763ade-3824-4ff6-9b05-3bebc7671e6e)

![image](https://github.com/mertmetin1/heart-attack-complications-classification-/assets/98667673/0a033131-3186-434a-a609-33669ebbdb23)

**7. Recommendations**

Based on our findings, we recommend deploying the Random Forest classifier in real-world applications for activity recognition using accelerometer data. Future work could focus on exploring additional feature selection methods and optimizing the classification models further to improve their performance.

