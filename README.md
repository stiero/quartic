# Quartic 

##### Notes 

I set aside a portion of my machine's disk space as swap memory to train and validate several models. Since this is a big dataset and preprocessing made it even bigger, I had issues with RAM, with tasks often stopping due to memory exhaustion. 


##### Dependencies
I ran the project on Ubuntu 16.04 LTS.

I use the Miniconda Python distribution on my machine, and so I created a separate conda virtual environment for this project (clone file included). 

To clone an identical environment in your local setup, navigate to the repository and run 

```sh
conda create --name your_env_name --file quartic_env.txt
```

Then activate your environment by running

```sh
source activate your_env_name
```

### Assumptions
- Outcome value of '0' is a non-event, a value of '1' is an event.
- There is no ordinal relationship between the categories in any given categorical feature.
- The derived ('der') features are all numeric and not categorical.

### 1a) My approach

- I first started out with some exploratory data analysis. I was particularly interested in the missing values. Upon plotting the histogram for each numerical feature with missing and reading its summary statistics, in each case I visually conjectured that the median will probably be the best metric to impute from, since the odd outlier was present in some cases, and in others, the median and mean were pretty close. 

- With categorical features, often the absence of a value can be a useful indicator for the model to pick up on. This is why I left the missing values of categorical features intact before encoding them (more on this below).
- For encoding, I usde two strategies based on the number of unique values in the categorical feature. If that number is <= 2, label encoding is used. If > 2, one-hot encoding is used. This is because several algorithms treat nominal encoding (0,1,2,3..) as ordinal, although presumably such an ordinal relationship doesn't exist.  
- To make testing easier, I wrote a function for scaling numerical features. I also had a go at dimensionality reduction using PCA, for which there is another function. Both these are optional techniques. 
- During the initial EDA, I also found that there was a heavy class imbalance in the training data (~27:1 in favour of the non-event class). Although a lot of models support balancing of classes, I decided to remedy it by random resampling of observations marked as an event (with resampling) in order to increase the signal of the event class and appending them to the training data to create a balance.
- I did a train-test split of the training data for cross-validation.
- For training, I started out with a Logistic Regression model, then one of Multinomial Naive Bayes, then KNN, SVC and tree-based models. Basically, I attempted to start with simpler and faster models and then move on to more complex and computationally expensive ones. 
- All through, I tried to implement code reusability and modularity as far as possible. 

#### 1b) Tradeoffs
- Because of the one-hot encoding of categorical features, the number of features greatly increased from 56 to 203. In trying to model the data more accurately, sparisity was introduced. This can potentially cause problems due to the so-called curse of dimensionality. 

- During the training process, I had to strike a balance between the model sensitivity and specificity, often by altering the threshold probabilty. As I do not have information about the cost or significance missclassification, I operated under the assumption that incorrectly classifying a non-event as an event was a lesser evil than incorrectly classifying an event as a non-event. This is why I tried to prioritise the model sensitivity, but this also came with the cost of decreased model specificity and potentially reducing the overall ROC AUC metric. 


### 2) Model details

- After trying out a few models ranging from the simplest to increasingly complex, I found that the XGBoost classifier offered the best performance with a reasonable computational expense. 

- After running a (very) lengthy grid search over several nights, the best individual model I could train had an ROC AUC of 0.62. I used AUC as the primary metric to gauge model performance because plain accuracy is blind to the sensitivity-specificity tradeoff. 
- For the final majority voting classification, I trained a total of 6 classification models. They include Logistic regression, Random Forest, XGBoost, Adaptive Boost, Light Gradient Boosting Machines and Multilayer Perceptron neural network. 
- Through majority voting, I finally arrived at the following metrics on the validation split -

| Metric | Value |
| ------ | ------ |
| Accuracy | 73.4% |
| ROC AUC | 0.73 |
| Cohen's kappa | 0.46 |
| Sensitivity | 0.85 |
| Specificity |  0.61 |

### 3) Further improvements
- If I had more time, I would first try and do a more thorough grid search for the optimal parameters for the models already trained to increase overall accuracy. 

- I would also experiment with some more feature engineering to more accurately capture the patterns in the data. Perhaps try a few polynomial combination of numerical features to see how the models behave, to see if there is an interaction effect between them. I have also read reports from a few practitioners about generating new features by building clustering models of the existing data.

- Another thing on my list would be stacking different models, ie. adding the response of one model as an input feature to train the next model. I prepared a script for this, but running it takes a very long time, since each model must be trained with a k-fold cross validation and these models each have to be tuned first to perform well. 



