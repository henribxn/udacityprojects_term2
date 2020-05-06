# Term 2 projects of the Udacity Data Science nanodegree program
Henri Bouxin
Last published Date: 2020 May 5th

## Introduction
The goal is to use the training data to understand what patterns in V1-V7 indicate that a promotion should be provided to a user. Specifically, the goal is to maximize the following metrics:
- `Incremental Response Rate (IRR)` : Ratio of the number of purchasers in the promotion group to the total number of customers in the purchasers group minus the ratio of the number of purchasers in the non-promotional group to the total number of customers in the non-promotional group.
- `Net Incremental Revenue (NIR)`: The total number of purchasers that received the promotion times 10 minus the number of promotions given times 0.15 minus the number of purchasers who were not given the promotion times 10.

## Repo structure
- **data** : it contains the train and test data. Each data point includes one column indicating whether or not an individual was sent a promotion for the product, and one column indicating whether or not that individual eventually purchased that product. Each individual also has seven additional features associated with them, which are provided abstractly as V1-V7

- **Starbucks.iynb** : jupyter notebook with all the differents steps of the project with the following table of content :

- `I. Statistical significance testing of the AB test results`
- `II. Building an optimization strategy with a supervised learning algorithm`

## Key takeaways :
- Learn

## Licensing, Acknowledgements
- These project were completed as part of the second and last term of the `Udacity Data Scientist Nanodegree program`
