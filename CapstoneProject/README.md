# Starbuck's Capstone Challenge
Author: Henri Bouxin

Start Date: 2020 May 5th

## Project description and motivation
The goal of the project is to analyse Startbucks that contain simulated data mimicking customer behavior on the Starbucks rewards mobile app. Indeed, once every few days, Starbucks sends out an offer to users of the mobile app.

I focused my work on trying to answer the following question:

1. What is the impact of such offers in the company cash inflows ?
2. Can demographic features help us understand the interactions between users and offers ?
3. How to improve customer experience in terms of promotional offers by personalizing the distribution process using a recommendation engine ?
4. Can we predict whether or not someone will respond to an offer ?

For example, many other questions could also have been answered using such a data set such as:
- Can we predict how much someone will spend based on demographics and offer type ?
- Can we predict whether or not someone will respond to an offer ?
- etc...


## Repo structure
- **data** :
  - Three starbucks json files (detailed in the data description section below)
  - Intermediate pickle file created during the heavy feature engineering part


- **Starbucks_Capstone_notebook.ipynb**: jupyter notebook with all the differents steps of the project with the following table of contents:
  - `I. Data Exploration and cleaning`
  - `II. Feature Engineering of Promotion Lifecyle`
  - `III. Data Vizualisation`
  - `IV. User-Item Collaborative Filtering with Funk SVD`

- **utils.py** : built-in functions to implement the heavy part of the feature engineering. In short, it turns the `transcript.json` file into a dataframe where each row is a user-offer pair and the columns give the details of the interactions between the user and the sent offer: what is viewed? What is completed? What is the spent amount related to such an offer?

## Data description
The data is contained in three files:
- `portfolio.json` - containing offer ids and meta data about each offer (duration, type, etc.)
- `profile.json` - demographic data for each customer
- `transcript.json` - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:
`portfolio.json`
- id (string) - offer id
- offer_type (string) - type of offer ie BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings)

`profile.json`
- age (int) - age of the customer
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

`transcript.json`
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record

## Package Used
This project requires Python 3.x and the following Python libraries installed:
- NumPy
- Pandas
- Matplotlib
- Json
- Seaborn
- Math
- Time

You will also need to have software installed to run and execute an iPython Notebook

## Conclusion and further improvements
- `Key takeaways`: end-to-end implementation of a recommendation engine combining both user based `Collaborative filtering with a FunkSVD` implementation and a content based approach to deal with the Cold Start Problem
- `Further Improvements`:


## Blog Post
You can also find more details on the project by following the link to my blogpost: `XX`

## License
This project was completed as part of the `Udacity Data Scientist Nanodegree`. Data were provided by Udacity. The data was originally sourced by Udacity from Starbucks.
