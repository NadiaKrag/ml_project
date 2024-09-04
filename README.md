# Notes
This markdown file is dedicated to thoughts, notes, and reflections.

## Potential models
Candidates for the first and third model in our report.

### Linear classification
- Simple linear discriminant (pg. 181-184)
- Least-squares for classification (pg. 184-186)
- Perceptron (pg. 192-196)
- Probabilistic generative model (pg. 196-203)
- (Bayesian) logistic regression (pg. 217-220)

### Non-linear classification
- Neural networks (pg. 225-231)
- Support vector machines (pg. 325-336)
- Gaussian mixture model (pg. 110-113 and 435-439)

### Other
- Kernel methods in conjunction with models
- Semi-supervised clustering

# Log book

**Monday, November 26th:**\
Torp looks at logistic regression with basis functions.\
Maher looks at standardization of the data and fixing the singular matrix problem with MDA.\
Nadia looks at various ways to improve AdaBoost itself.\
Stefan looks at how AdaBoost and MDA compares.

**Wednesday, November 21st:**\
Firstly, we figured out that the latent variable formulation of the MDA seemed correct according to Pierre. Secondly, we discussed various possibilities with AdaBoost and that the weakest of the weak learners might be the way to go (decision stumps).

Viktor talked about using the clustering from MDA or GMM to further aid other models as well as AdaBoost.\
Nadia wanted to keep working on AdaBoost.\
Stefan talked about looking more into weak learners and the linear model.

**Monday, November 19th:**\
The GMM is implemented. However, we are not sure it is a linear classification model. Regarding the AdaBoost model we talked about trying out different classifiers and different ways to optimise it. We also talk about the LVM formulation of MDA, but we are not quite sure we are on the right track. We have agreed to meet Wednesday 13.00 before the exercise lessons.

Maher focus on the Gaussian mixture model\
Viktor focus on implementation of AdaBoost with a Linear Regression based classifier (link below*)\
Nadia focus on implementation of AdaBoost with a "decision stump" based classifier\
Stefan relaxes and focus on the LVM formulation of MDA

Questions for exercise lessons:
- GMM, a linear classification model?
- MDA latent variable model

*https://anujkatiyal.com/blog/2017/10/24/ml-adaboost/?fbclid=IwAR3gttlrKakhTxSj9bOTOdxkmkmPecVqKjtLPZ8asU4Yvrpn2gaarrIn0M0#.W_QQn5NKifV

**Wednesday, November 14th:**\
We went through the book to look for potential linear and non-linear models for the first and third project model respectively. The frontrunner for the linear model is the Gaussian mixture model as a comparison to the MDA. For the third model, we decided on an ensembling method - this has the benefit of going through the last chapter of the book (chapter 14), as well as generally being considered damn good. For now the AdaBoost algorithm is considered.\
We also went through the MDA model, and conjectured that using a kernel function could aid the model.

Maher wanted to focus on the Gaussian mixture model (first model for the project).\
Nadia wanted to focus on the AdaBoost algorithm (third model for the project).\
Stefan wanted to focus on the LVM formulation of MDA.

**Monday, November 12th:**\
The project and data were introduced. We examined the data with a few simple plots, and arranged further meetings.
