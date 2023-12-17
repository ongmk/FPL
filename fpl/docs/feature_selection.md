- **Low correlation features**
  Although some features have low correlation with FPL points, they could still contribute to the predictive
  power of the model in combination with other features, so they are kept.

- **MA5**
  Moving averages with a lag window of 5 seem to be the most correlated to FPL points. For consistency, MA5 is used for
  all the features.

- **Home ELO vs ELO**
  Home/away performances and overall performances are equally important to capture. But using both might cause the problem
  of multicollinearity. Although tree models are less affected by multicollinearity, there are different ways to prevent it:
  - Feature engineering
    Calculate alternative features that capture relevant information, e.g. difference between home and away games
  - Dimensionality reduction (PCA)
  - Regularization (Ridge or LASSO)
    Introducing penalty terms can shrink the coefficients of correlated features.
