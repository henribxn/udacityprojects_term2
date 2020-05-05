import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time

def extract_promo_information_and_related_transactions(transcript, user_id, portfolio):
    '''
    Goal: for a given person, understand the different promotion lifecyle and how they are related to existing transactions
    Input:
    - transcript (dataframe): dataframe of the transactions
    - user_id (string): person to be considered
    - portfolio
    Output:
    - offer_life_cycle_final (dataframe): for each cretaed promotion, follow its lifecycle by indicating if it has been viewed, and/or completed
    - transaction_id_full (dataframe): same dataframe as transcript, enriched by the associated promotion of each transaction if it exists

    '''
    # Get the transaction of a given user_id
    transaction_id = transcript[transcript.person ==user_id].drop(labels=['value'],axis=1)

    # Merge with transaction informations
    transaction_id_full = pd.merge(transaction_id,portfolio, left_on="offer_id",right_on="id", how="left").drop(labels=["channels","id"],axis=1)

    # Enrich the transaction table with connected promotions and their respective information 
    transaction_id_full["linked_promo"]=0 # 0 by default, 1 if a promotion has been seen and has been followed by a transaction in the duration period
    transaction_id_full["creation_time"]=0
    transaction_id_full["expiration_time"]=0

    # Instanciate the three dataframe following promotion lifecycles
    dict_time_offer_received =pd.DataFrame({
        "offer_id":[],
        "creation_time":[],
        "expiration_time":[]
    })

    dict_time_offer_viewed =pd.DataFrame({
        "offer_id":[],
        "creation_time":[],
        "time":[],
        "expiration_time":[]
    })

    dict_time_offer_completed =pd.DataFrame({
        "offer_id":[],
        "time":[],
        "expiration_time":[]
    })

    # Instantiate the liste to follow the promotion that have been seen and completed already
    offer_completed =[]

    # Iteration over each transaction line
    for i,v in transaction_id_full.iterrows():

         # Append the dict_time_offer_received dataframe with the offer received information
        if v["event"]=="offer received":
            dict_time_offer_received = dict_time_offer_received.append({'offer_id':v["offer_id"],'creation_time':int(v["time"]),"expiration_time":int(v["time"]+int(v["duration"]*24))},ignore_index=True)
        
        # Append the dict_time_offer_received dataframe with the offer viewed information
        if v["event"]=="offer viewed":
            
            offer_id = v["offer_id"]
            creation_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["creation_time"].iloc[-1])
            expiration_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["expiration_time"].iloc[-1])
            dict_time_offer_viewed = dict_time_offer_viewed.append({'offer_id':offer_id,'creation_time':creation_time,"time":v["time"],"expiration_time":expiration_time},ignore_index=True)

            # Rule: To take into account the case when an offer is repushed and reviewed, we need to reset the offer_completed liste (and therefore to erase the corresponding value in offer_completed)
            if offer_id in offer_completed :
                offer_completed = np.setdiff1d(offer_completed,offer_id,assume_unique=True).tolist()
        
        # Append the dict_time_offer_received dataframe with the offer completed information
        if v["event"]=="offer completed":
            offer_id = v["offer_id"]
            creation_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["creation_time"].iloc[-1])
            expiration_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["expiration_time"].iloc[-1])
            dict_time_offer_completed = dict_time_offer_completed.append({'offer_id':offer_id,'creation_time':creation_time,"time":v["time"],"expiration_time":expiration_time},ignore_index=True)

            # Append the liste with the completed offer
            offer_completed.append(v["offer_id"])
        
         # Affect promotions that have been seen and not completed yet to transactions
        if v["event"]=="transaction":
            # Rule 1: An promotion is affected to a transaction ("is_trans_promo_driven") only if it has not been completed yet.
            # Rule 2: If two promotions have been viewed and not completed yet, we consider that only the last transaction
            
            # Check that dict_time_offer_viewed is non null
            if dict_time_offer_viewed.shape[0]>0 :
                is_transaction_promotion_driven = dict_time_offer_viewed[(dict_time_offer_viewed["expiration_time"]>= int(v["time"])) & (int(v["time"])>=dict_time_offer_viewed["creation_time"])]
                
                #Check that is_transaction_promotion_driven is non null
                if is_transaction_promotion_driven.shape[0]>0:
                    is_transaction_promotion_driven_without_completed = np.setdiff1d(is_transaction_promotion_driven.offer_id,offer_completed,assume_unique=True).tolist()
                    if len(is_transaction_promotion_driven_without_completed)>0:
                        linked_promo_id = is_transaction_promotion_driven_without_completed[-1]
                        transaction_id_full.loc[i,"linked_promo"]= linked_promo_id
                        transaction_id_full.loc[i,"creation_time"]= is_transaction_promotion_driven[is_transaction_promotion_driven.offer_id ==linked_promo_id]["creation_time"].iloc[-1]
                        transaction_id_full.loc[i,"expiration_time"]= is_transaction_promotion_driven[is_transaction_promotion_driven.offer_id ==linked_promo_id]["expiration_time"].iloc[-1]
            
                
    # Build the life cycle of every promotion using the unique key [offer_id,creation_time,expiration_time]
    dict_time_offer_viewed["is_viewed"]=1
    dict_time_offer_completed["is_completed"]=1

    # Distinct all the different cases
   
    if (dict_time_offer_received.shape[0]>0)&(dict_time_offer_viewed.shape[0]>0)&(dict_time_offer_completed.shape[0]>0):
        offer_life_cycle1 = pd.merge(dict_time_offer_received,dict_time_offer_viewed,on=["offer_id","creation_time","expiration_time"],how='left').drop(labels=["time"],axis=1)
        offer_life_cycle_final = pd.merge(offer_life_cycle1,dict_time_offer_completed,on=["offer_id","creation_time","expiration_time"],how='left').drop(labels=["time"],axis=1)           
    else :
        offer_life_cycle_final = pd.DataFrame()
    
    return offer_life_cycle_final,transaction_id_full





def final_promotion_lifecycle(offer_life_cycle_final,transaction_id_full,user_id):
    '''
    For a given person, get the final lifecycle dataframe building the lifecycle of every promotion and the associated amount spent
    Input :
    - user_id (string): person to be considered
    - offer_life_cycle_final (dataframe): for each cretaed promotion, follow its lifecycle by indicating if it has been viewed, and/or completed
    - transaction_id_full (dataframe): same dataframe as transcript, enriched by the associated promotion of each transaction if it exists
    Output :
    - offer_life_cycle_final_all: same as offer_life_cycle_final, enriched by the transaction amount (mean and total), and append with the 0 promotion
    
    '''
    # Extract relevant columns from the transaction_id_full dataframe
    transaction_id_full_extract = transaction_id_full[["amount_spent","linked_promo","creation_time","expiration_time"]].rename(columns= {"linked_promo":"offer_id"})
    
    # Consolidate average amount and total amount spent for each promotion (the promotion equal to 0 being the one not related to any promotions)
    transaction_id_full_extract_mean = transaction_id_full_extract.groupby(["offer_id","creation_time","expiration_time"])["amount_spent"].mean()
    transaction_id_full_extract_sum = transaction_id_full_extract.groupby(["offer_id","creation_time","expiration_time"])["amount_spent"].sum()
    
    transaction_id_full_extract_mean1= pd.DataFrame(transaction_id_full_extract_mean).reset_index().rename(columns={"amount_spent":"mean_amount_spent"})
    transaction_id_full_extract_sum1= pd.DataFrame(transaction_id_full_extract_sum).reset_index().rename(columns={"amount_spent":"total_amount_spent"})
    transaction_id_full_extract_amount_spent = pd.merge(transaction_id_full_extract_mean1,transaction_id_full_extract_sum1,on=["offer_id","creation_time","expiration_time"],how="inner")
   
    # Append the 0 promotion (i.e. the one not related to any promotion) to the lifecyle promotion
    offer_life_cycle_final_1 = offer_life_cycle_final.append({'offer_id':0,'creation_time':0.0,"expiration_time":0.0,"is_viewed":np.nan,"is_completed":np.nan},ignore_index=True)
   
    # Merge the lifecycle promotion dataframe with the corresponding amount transaction
    try:
        offer_life_cycle_final_all =pd.merge(offer_life_cycle_final_1,transaction_id_full_extract_amount_spent,on=["offer_id","creation_time","expiration_time"],how='left')
    except ValueError:
        offer_life_cycle_final_all = pd.DataFrame()
    # Check that all the amounts are correct
    ## assert abs((offer_life_cycle_final_all.total_amount_spent.sum()-transaction_id_full.amount_spent.sum())/transaction_id_full.amount_spent.sum())<0.10,"Oops there is a problem"
    
    return offer_life_cycle_final_all


def building_user_promotion_cycle(transcript,portfolio,user_id_init,limit):
    '''
    Loop over all the unique person in the transcript table in order to build the promotion cycle, 
    i.e. for each received offer, know if it has been viewed, completed, and the associated spent amount
    INPUT : 
    - transacript (dataframe): transcript table
    -portfolio (dataframe): portfolio table
    - user_id_init (int): user_id that has been used to initialize the user_promotion_cycle dataframe
    - limit (int): exit the function when the number of looped unique person is greater than such a limit, for testing only
    OUTPUT : 
    - user_promotion_cycle (dataframe): dataframe with the promotion cycle for every user
    - non_working_user_id (list) : list of the users for whom the built-in functions raises exceptions
    '''
    
    user_promotion_cycle = pd.DataFrame()
    counter = 1
    start = time.time()
    non_working_user_id=[]
    for user in list(set(transcript.person)):
        if counter > limit :
            break
        if user != user_id_init: 
            offer_life_cycle_final,transaction_id_full = extract_promo_information_and_related_transactions(transcript,user,portfolio)
            if offer_life_cycle_final.shape[0]>0:
                offer_life_cycle_final_all = final_promotion_lifecycle(offer_life_cycle_final,transaction_id_full,user)
                offer_life_cycle_final_all["user_id"]=user
                user_promotion_cycle = pd.concat([user_promotion_cycle,offer_life_cycle_final_all])

            else :
                non_working_user_id.append(user)
                continue

            counter = counter+1
            if counter % 100 ==0:
                end = time.time()-start
                print("It takes {} mins to make {} batch of iterations".format(end/60,counter))
    return user_promotion_cycle,non_working_user_id


def extract_promo_information_and_related_transactions_new(transcript,user_id,portfolio):
    '''
    Goal: for a given person, understand the different promotion lifecyle and how they are related to existing transactions
    Input:
    - transcript (dataframe): dataframe of the transactions
    - user_id (string): person to be considered
    Output:
    - offer_life_cycle_final (dataframe): for each cretaed promotion, follow its lifecycle by indicating if it has been viewed, and/or completed
    - transaction_id_full (dataframe): same dataframe as transcript, enriched by the associated promotion of each transaction if it exists

    '''
    # Get the transaction of a given user_id
    transaction_id = transcript[transcript.person ==user_id].drop(labels=['value'],axis=1)

    # Merge with transaction informations
    transaction_id_full = pd.merge(transaction_id,portfolio, left_on="offer_id",right_on="id", how="left").drop(labels=["channels","id"],axis=1)

    # Enrich the transaction table with connected promotions and their respective information 
    transaction_id_full["linked_promo"]=0 # 0 by default, 1 if a promotion has been seen and has been followed by a transaction in the duration period
    transaction_id_full["creation_time"]=0
    transaction_id_full["expiration_time"]=0

    # Instanciate the three dataframe following promotion lifecycles
    dict_time_offer_received =pd.DataFrame({
        "offer_id":[],
        "creation_time":[],
        "expiration_time":[]
    })

    dict_time_offer_viewed =pd.DataFrame({
        "offer_id":[],
        "creation_time":[],
        "time":[],
        "expiration_time":[]
    })

    dict_time_offer_completed =pd.DataFrame({
        "offer_id":[],
        "time":[],
        "expiration_time":[]
    })

    # Instantiate the liste to follow the promotion that have been seen and completed already
    offer_completed =[]

    # Iteration over each transaction line
    for i,v in transaction_id_full.iterrows():

         # Append the dict_time_offer_received dataframe with the offer received information
        if v["event"]=="offer received":
            dict_time_offer_received = dict_time_offer_received.append({'offer_id':v["offer_id"],'creation_time':int(v["time"]),"expiration_time":int(v["time"]+int(v["duration"]*24))},ignore_index=True)
        
        # Append the dict_time_offer_received dataframe with the offer viewed information
        if v["event"]=="offer viewed":
            
            offer_id = v["offer_id"]
            creation_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["creation_time"].iloc[-1])
            expiration_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["expiration_time"].iloc[-1])
            dict_time_offer_viewed = dict_time_offer_viewed.append({'offer_id':offer_id,'creation_time':creation_time,"time":int(v["time"]),"expiration_time":expiration_time},ignore_index=True)

            # Rule: To take into account the case when an offer is repushed and reviewed, we need to reset the offer_completed liste (and therefore to erase the corresponding value in offer_completed)
            if offer_id in offer_completed :
                offer_completed = np.setdiff1d(offer_completed,offer_id,assume_unique=True).tolist()
        
        # Append the dict_time_offer_received dataframe with the offer completed information
        if v["event"]=="offer completed":
            offer_id = v["offer_id"]
            creation_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["creation_time"].iloc[-1])
            expiration_time = int(dict_time_offer_received[dict_time_offer_received.offer_id==offer_id]["expiration_time"].iloc[-1])
            dict_time_offer_completed = dict_time_offer_completed.append({'offer_id':offer_id,'creation_time':creation_time,"time":int(v["time"]),"expiration_time":expiration_time},ignore_index=True)

            # Append the liste with the completed offer
            offer_completed.append(v["offer_id"])
        
         # Affect promotions that have been seen and not completed yet to transactions
        if v["event"]=="transaction":
            # Rule 1: An promotion is affected to a transaction ("is_trans_promo_driven") only if it has not been completed yet.
            # Rule 2: If two promotions have been viewed and not completed yet, we consider that only the last transaction
            
            # Check that dict_time_offer_viewed is non null
            if dict_time_offer_viewed.shape[0]>0 :
                is_transaction_promotion_driven = dict_time_offer_viewed[(dict_time_offer_viewed["expiration_time"]>= v["time"]) & (v["time"]>=dict_time_offer_viewed["creation_time"])]
                
                #Check that is_transaction_promotion_driven is non null
                if is_transaction_promotion_driven.shape[0]>0:
                    is_transaction_promotion_driven_without_completed = np.setdiff1d(is_transaction_promotion_driven.offer_id,offer_completed,assume_unique=True).tolist()
                    if len(is_transaction_promotion_driven_without_completed)>0:
                        linked_promo_id = is_transaction_promotion_driven_without_completed[-1]
                        transaction_id_full.loc[i,"linked_promo"]= linked_promo_id
                        transaction_id_full.loc[i,"creation_time"]= int(is_transaction_promotion_driven[is_transaction_promotion_driven.offer_id ==linked_promo_id]["creation_time"].iloc[-1])
                        transaction_id_full.loc[i,"expiration_time"]= int(is_transaction_promotion_driven[is_transaction_promotion_driven.offer_id ==linked_promo_id]["expiration_time"].iloc[-1])
            
                
    # Build the life cycle of every promotion using the unique key [offer_id,creation_time,expiration_time]
    dict_time_offer_viewed["is_viewed"]=1
    dict_time_offer_completed["is_completed"]=1

    # Distinct all the different cases
    if (dict_time_offer_received.shape[0]>0):
        if (dict_time_offer_viewed.shape[0]>0):
            offer_life_cycle1 = pd.merge(dict_time_offer_received,dict_time_offer_viewed,on=["offer_id","creation_time","expiration_time"],how='left').drop(labels=["time"],axis=1)
        if (dict_time_offer_completed.shape[0]>0) :
            offer_life_cycle_2 = pd.merge(dict_time_offer_received,dict_time_offer_completed,on=["offer_id","creation_time","expiration_time"],how='left').drop(labels=["time"],axis=1)           
        if (dict_time_offer_viewed.shape[0]>0) & (dict_time_offer_completed.shape[0]>0):    
            offer_life_cycle_final = pd.merge(offer_life_cycle1,offer_life_cycle_2,on=["offer_id","creation_time","expiration_time"],how='outer').drop(labels=["time"],axis=1)           

    else :
        offer_life_cycle_final = pd.DataFrame()
        
    return offer_life_cycle_final,transaction_id_full