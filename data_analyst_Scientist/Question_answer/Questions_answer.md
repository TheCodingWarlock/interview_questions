# Data Science/Analytics Questions Interview Questions

Common Data Science questions curated from the internet.<br>
*Disclaimer* I'm not in HR.<br>
Sources
1. [edureka](https://www.edureka.co/blog/interview-questions/top-data-science-interview-questions-for-budding-data-scientists/)

<br>

<details><summary><b>What is Data Science? Also, list the differences between supervised and unsupervised learning. </b></summary>

 > Data science is a multi-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.

 > supervised learning
   * Input data is labeled.
   * Uses training dataset.
   * Used for prediction.
   * Enables classification and regression.
> Unsupervised Learning
   * Input data is unlabeled.
   * Uses the input data set.
   * Used for analysis.
   * Enables Classification, Density Estimation, & Dimension Reduction

 > Supervised learning: Supervised learning is the learning of the model where with input variable ( say, x) and an output variable (say, Y) and an algorithm to map the input to the output.
That is, Y = f(X) 

 > Unsupervised learning is where only the input data (say, X) is present and no corresponding output variable is there.


</details>


<details><summary><b> What are the important skills to have in Python with regard to data analysis? </b></summary>

 >  * Good understanding of the built-in data types especially lists, dictionaries, tuples, and sets.
    * Mastery of N-dimensional NumPy Arrays.
    * Mastery of Pandas dataframes.
    * Ability to perform element-wise vector and matrix operations on NumPy arrays.
    * Knowing that you should use the Anaconda distribution and the conda package manager.
    * Familiarity with Scikit-learn. **Scikit-Learn Cheat Sheet**
    * Ability to write efficient list comprehensions instead of traditional for loops.
    * Ability to write small, clean functions (important for any developer), preferably pure functions that don’t alter objects.
    * Knowing how to profile the performance of a Python script and how to optimize bottlenecks.



</details>

<details><summary><b>  What is Selection Bias? </b></summary>

 > Selection bias is a kind of error that occurs when the researcher decides who is going to be studied. It is usually associated with research where the selection of participants isn’t random. It is sometimes referred to as the selection effect. It is the distortion of statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may not be accurate.

 > The types of selection bias include:

    Sampling bias: It is a systematic error due to a non-random sample of a population causing some members of the population to be less likely to be included than others resulting in a biased sample.

    Time interval: A trial may be terminated early at an extreme value (often for ethical reasons), but the extreme value is likely to be reached by the variable with the largest variance, even if all variables have a similar mean.

    Data: When specific subsets of data are chosen to support a conclusion or rejection of bad data on arbitrary grounds, instead of according to previously stated or generally agreed criteria.

    Attrition: Attrition bias is a kind of selection bias caused by attrition (loss of participants) discounting trial subjects/tests that did not run to completion.

</details>


<details><summary><b> What is the difference between “long” and “wide” format data? </b></summary>

 > In the wide format, a subject’s repeated responses will be in a single row, and each response is in a separate column. In the long format, each row is a one-time point per subject. You can recognize data in wide format by the fact that columns generally represent groups.

</details>


<details><summary><b> What do you understand by the term Normal Distribution? </b></summary>

 > Data is usually distributed in different ways with a bias to the left or to the right or it can all be jumbled up.

    However, there are chances that data is distributed around a central value without any bias to the left or right and reaches normal distribution in the form of a bell-shaped curve.
    The random variables are distributed in the form of a symmetrical bell-shaped curve.

    Properties of Nornal Distribution:

    Unimodal -one mode
    Symmetrical -left and right halves are mirror images
    Bell-shaped -maximum height (mode) at the mean
    Mean, Mode, and Median are all located in the center
    Asymptotic

</details>

<details><summary><b> What is the goal of A/B Testing? </b></summary>

 > It is a statistical hypothesis testing for a randomized experiment with two variables A and B.

    The goal of A/B Testing is to identify any changes to the web page to maximize or increase the outcome of an interest. A/B testing is a fantastic method for figuring out the best online promotional and marketing strategies for your business. It can be used to test everything from website copy to sales emails to search ads

</details>


<details><summary><b> What do you understand by statistical power of sensitivity and how do you calculate it? </b></summary>

 > Sensitivity is commonly used to validate the accuracy of a classifier (Logistic, SVM, Random Forest etc.).

    Sensitivity is nothing but “Predicted True events/ Total events”. True events here are the events which were true and model also predicted them as true.

    Calculation of seasonality 

    Seasonality = ( True Positives ) / ( Positives in Actual Dependent Variable )

</details>


<details><summary><b> What are the differences between overfitting and underfitting? </b></summary>

 > Overfitting occurs when a statistical model or machine learning algorithm captures the noise of the data.  Intuitively, overfitting occurs when the model or the algorithm fits the data too well.  Specifically, overfitting occurs if the model or algorithm shows low bias but high variance.  Overfitting is often a result of an excessively complicated model, and it can be prevented by fitting multiple models and using validation or cross-validation to compare their predictive accuracies on test data.

 > Underfitting occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data.  Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough.  Specifically, underfitting occurs if the model or algorithm shows low variance but high bias.  Underfitting is often a result of an excessively simple model.

</details>

<details><summary><b>  Python or R – Which one would you prefer for text analytics? </b></summary>

 > We will prefer Python because of the following reasons:

    Python would be the best option because it has Pandas library that provides easy to use data structures and high-performance data analysis tools.
    R is more suitable for machine learning than just text analysis.
    Python performs faster for all types of text analytics.

</details>

<details><summary><b> How does data cleaning plays a vital role in analysis? </b></summary>

 > Data cleaning can help in analysis because:

    Cleaning data from multiple sources helps to transform it into a format that data analysts or data scientists can work with.
    Data Cleaning helps to increase the accuracy of the model in machine learning.
    It is a cumbersome process because as the number of data sources increases, the time taken to clean the data increases exponentially due to the number of sources and the volume of data generated by these sources.
    It might take up to 80% of the time for just cleaning data making it a critical part of analysis task.

</details>

<details><summary><b> Differentiate between univariate, bivariate and multivariate analysis. </b></summary>

 > Univariate analyses are descriptive statistical analysis techniques which can be differentiated based on the number of variables involved at a given point of time. For example, the pie charts of sales based on territory involve only one variable and can the analysis can be referred to as univariate analysis.

> The bivariate analysis attempts to understand the difference between two variables at a time as in a scatterplot. For example, analyzing the volume of sale and spending can be considered as an example of bivariate analysis.

> Multivariate analysis deals with the study of more than two variables to understand the effect of variables on the responses.

</details>


<details><summary><b> What is Cluster Sampling? </b></summary>

 > Cluster sampling refers to a type of sampling method . With cluster sampling, the researcher divides the population into separate groups, called clusters. Then, a simple random sample of clusters is selected from the population. The researcher conducts his analysis on data from the sampled clusters.

</details>


<details><summary><b> What is Systematic Sampling? </b></summary>

 > Systematic sampling is a statistical technique where elements are selected from an ordered sampling frame. In systematic sampling, the list is progressed in a circular manner so once you reach the end of the list, it is progressed from the top again. The best example of systematic sampling is equal probability method.

</details>


<details><summary><b> What are Eigenvectors and Eigenvalues? </b></summary>

 > Eigenvectors are used for understanding linear transformations. In data analysis, we usually calculate the eigenvectors for a correlation or covariance matrix. Eigenvectors are the directions along which a particular linear transformation acts by flipping, compressing or stretching.

> Eigenvalue can be referred to as the strength of the transformation in the direction of eigenvector or the factor by which the compression occurs.

</details>

<details><summary><b> Can you cite some examples where a false positive is important than a false negative? </b></summary>

 >      False Positives are the cases where you wrongly classified a non-event as an event a.k.a Type I error.
    False Negatives are the cases where you wrongly classify events as non-events, a.k.a Type II error.

> A false positive is where you receive a positive result for a test, when you should have received a negative results
> False Negatives, you get a negative test result, but you should have got a positive test result.

</details>


<details><summary><b> Can you cite some examples where both false positive and false negatives are equally important? </b></summary>

 > false positives:

    A pregnancy test is positive, when in fact you aren’t pregnant.
    A cancer screening test comes back positive, but you don’t have the disease.
    A prenatal test comes back positive for Down’s Syndrome, when your fetus does not have the disorder(1).
    Virus software on your computer incorrectly identifies a harmless program as a malicious one.

> False Negative
    Quality control in manufacturing; a false negative in this area means that a defective item passes through the cracks.
    In software testing, a false negative would mean that a test designed to catch something (i.e. a virus) has failed.
    In the Justice System, a false negative occurs when a guilty suspect is found “Not Guilty” and allowed to walk free.



</details>


<details><summary><b>Can you explain the difference between a Validation Set and a Test Set? </b></summary>

 > A Validation set can be considered as a part of the training set as it is used for parameter selection and to avoid overfitting of the model being built. On the other hand, a Test Set is used for testing or evaluating the performance of a trained machine learning model.

</details>


<details><summary><b> Explain cross-validation. </b></summary>

 > Cross-validation is a model validation technique for evaluating how the outcomes of statistical analysis will generalize to an Independent dataset. Mainly used in backgrounds where the objective is forecast and one wants to estimate how accurately a model will accomplish in practice. The goal of cross-validation is to term a data set to test the model in the training phase (i.e. validation data set) in order to limit problems like overfitting and get an insight on how the model will generalize to an independent data set.

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>



<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>



<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>



<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>



<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>