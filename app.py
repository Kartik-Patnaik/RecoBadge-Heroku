import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
@app.route('/',methods = ['GET', 'POST'])
def upload_file():
   return render_template('index.html')
@app.route('/data',methods = ['GET', 'POST'])
def data():
    if request.method=="POST":
        f = request.form['csvfile']
        data = pd.read_csv(f)
        dataset = data.drop(["EmployeeStatusDesc","Domain","Sub Domain","Badge Type","Badge Status","Key-Badge_SD","Badge earned","Initiate a badge date","Time Lapse","Badge Classification"],axis = 1)
        X = dataset.iloc[:, 1:4]
        #Create dummy variables
        Country=pd.get_dummies(X['Country'])
        Department=pd.get_dummies(X['Department'])
        Designation=pd.get_dummies(X['Designation'])

        ## Concatenate the Data Frames
        X=pd.concat([X,Country,Designation,Department],axis=1)
        ## Drop Unnecessary columns
        X=X.drop(['Country','Designation','Department'],axis=1)
        df = pd.read_pickle('saved_data.pkl')
        df = df.append(X, ignore_index=True)
        df = df.iloc[:, 0:48]
        df = df.fillna(0)
        # Feature Scaling
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        X1 = scaler.transform(df)
        #X1 = sc.transform(X)
        loaded_model = pickle.load(open('nn_classifier.pkl', 'rb'))
        predicted = loaded_model.predict(X1)
        #predicted = classifier.predict(X1)
        encoder = pickle.load(open('encoder.pkl', 'rb'))
        naming = encoder.inverse_transform(range(91))
        predicted_1 = pd.DataFrame(predicted)
        predicted_1.columns = naming
        predicted_2 = loaded_model.predict_classes(X1)
        predicted_2 = pd.DataFrame(encoder.inverse_transform(predicted_2))
        predicted_12 = pd.DataFrame(predicted_2)
        predicted_12.columns = ["predicted_class"]
        concatinated = pd.concat([dataset,predicted_12,predicted_1], axis=1)
        concatinated = pd.concat([dataset,predicted_12,predicted_1], axis=1)
        #data = concatinated
        # Extract GPN and Probability scores

        # import data
        model_output = concatinated
        source_data = pd.read_pickle('root_data.pkl')
        # Extract GPN and Probability scores
        GPN_ProbScore = model_output.drop(["Country","Designation","Department","predicted_class"],axis = 1)
        #########################################FIND ALTERNATE SOLUTION - Dynamic pick of col#############################################
        GPN_ProbScore_t = GPN_ProbScore.melt(id_vars = ['Requestor GPN'],
                                             value_vars = ['Analytics_Data architecture_Bronze','Analytics_Data architecture_Gold','Analytics_Data architecture_Platinum','Analytics_Data architecture_Silver','Analytics_Data integration_Bronze','Analytics_Data integration_Gold','Analytics_Data integration_Silver','Analytics_Data platform_Bronze','Analytics_Data platform_Gold','Analytics_Data platform_Silver','Analytics_Data science_Bronze','Analytics_Data science_Gold','Analytics_Data science_Platinum','Analytics_Data science_Silver','Analytics_Data visualization_Bronze','Analytics_Data visualization_Gold','Analytics_Data visualization_Platinum','Analytics_Data visualization_Silver','Analytics_Information strategy_Bronze','Analytics_Information strategy_Gold','Analytics_Information strategy_Silver','Cybersecurity_Cybersecurity_Bronze','Digital age teaming_Global team leadership_Bronze','Digital age teaming_Global team leadership_Silver','Digital age teaming_Inclusive Intelligence_Bronze','Digital age teaming_Inclusive Intelligence_Silver','Digital age teaming_Transformational leadership_Bronze','Digital_Digital_Bronze','Digital_Digital_Gold','Digital_Digital_Silver','Emerging Technology_Artificial intelligence_Bronze','Emerging Technology_Artificial intelligence_Gold','Emerging Technology_Artificial intelligence_Platinum','Emerging Technology_Artificial intelligence_Silver','Emerging Technology_Blockchain_Bronze','Emerging Technology_Blockchain_Gold','Emerging Technology_Blockchain_Platinum','Emerging Technology_Blockchain_Silver','Emerging Technology_Robotic process automation_Bronze','Emerging Technology_Robotic process automation_Gold','Emerging Technology_Robotic process automation_Platinum','Emerging Technology_Robotic process automation_Silver','Finance_Finance_Bronze','Innovation_Agile_Bronze','Innovation_Agile_Gold','Innovation_Agile_Silver','Innovation_Design thinking_Bronze','Innovation_Design thinking_Gold','Innovation_Design thinking_Platinum','Innovation_Design thinking_Silver','Microsoft_Dynamics 365_Bronze','Microsoft_Modern Workplace_Bronze','SAP_SAP Foundation_Bronze','Sector_Advanced Manufacturing_Bronze','Sector_BCM - Corporate & Commercial Banking_Bronze','Sector_BCM - Investment Banking & Capital Markets_Bronze','Sector_Consumer Products_Bronze','Sector_Government & Public Sector_Bronze','Sector_Insurance_Bronze','Sector_Life Sciences_Bronze','Sector_Mining & Metals_Bronze','Sector_Mobility_Bronze','Sector_Oil & Gas_Bronze','Sector_Oil & gas_Bronze','Sector_Power & Utilities_Bronze','Sector_Power & Utilities_Silver','Sector_Power & utilities_Bronze','Sector_Private Equity_Bronze','Sector_Real Estate, Hospitality & Construction_Bronze','Sector_Retail_Bronze','Sector_TMT - Media & Entertainment_Bronze','Sector_TMT - Technology_Bronze','Sector_TMT - Telecommunications_Bronze','Sector_TMT ? Media & Entertainment_Bronze','Sector_WAM - Alternative asset management_Bronze','Sector_WAM - Asset management_Bronze','Sector_WAM - Asset servicing_Bronze','Sector_WAM - Financial products_Bronze','Sector_WAM - Wealth management_Bronze','Strategy_Strategy_Bronze','Sustainability_Climate change and sustainability_Bronze','Transformative leadership_Agility_Bronze','Transformative leadership_Curiosity_Bronze','Transformative leadership_Inclusion and Belonging_Bronze','Transformative leadership_Inclusion and Belonging_Silver','Transformative leadership_Inspiring_Bronze','Transformative leadership_Inspiring_Silver','Transformative leadership_My Purpose_Bronze','Transformative leadership_Teaming_Bronze','Transformative leadership_Teaming_Silver','Transformative leadership_Wellbeing_Bronze'] ,
                                             var_name = "Reco Badges",
                                             value_name = "Prob Score").sort_values(by = ["Requestor GPN","Prob Score"],ascending = False)
        GPN_ProbScore_rank = GPN_ProbScore_t
        GPN_ProbScore_rank['Requestor GPN']=GPN_ProbScore_rank['Requestor GPN'].astype(str)
        GPN_ProbScore_rank['rank_initial'] = GPN_ProbScore_rank.groupby('Requestor GPN')['Prob Score'].rank(ascending=False)
        # Check past badges
        GPN_ProbScore_rank1 = GPN_ProbScore_rank
        GPN_ProbScore_rank1['key1'] =  GPN_ProbScore_rank1['Requestor GPN']+'_'+GPN_ProbScore_rank1["Reco Badges"].str.split("_", n = 2, expand = True)[1]
        source_data['Requestor GPN']=source_data['Requestor GPN'].astype(str)
        source_data['key1'] =  source_data['Requestor GPN']+'_'+source_data['Sub Domain']
        GPN_Reco = pd.merge(GPN_ProbScore_rank1,source_data,on = "key1", how = 'left')
        GPN_Reco['Reco badge type'] =  GPN_Reco["Reco Badges"].str.split("_", n = 2, expand = True)[2]
        # Remove badges which have already been approved or whose badge types are lower than approved ones
        GPN_Reco1 = GPN_Reco[(((GPN_Reco['Badge Type'] == "Bronze") &
                             ((GPN_Reco['Reco badge type'] == "Platinum") | (GPN_Reco['Reco badge type'] == "Gold") | (GPN_Reco['Reco badge type'] == "Silver"))) |
                            ((GPN_Reco['Badge Type'] == "Silver") &
                             ((GPN_Reco['Reco badge type'] == "Platinum") | (GPN_Reco['Reco badge type'] == "Gold"))) |
                              ((GPN_Reco['Badge Type'] == "Gold") &
                             ((GPN_Reco['Reco badge type'] == "Platinum"))) |
                             GPN_Reco['Country'].isnull())]
        GPN_Reco_Final = GPN_Reco1.iloc[:, 0:4].sort_values(by = ["Requestor GPN_x","Prob Score"],ascending = False)
        GPN_Reco_Final.rename(columns={"Requestor GPN_x":"Requestor GPN"}, inplace=True)
        # Pick 1st recommendation from sub domain
        GPN_Reco_Final['Domain'] = GPN_Reco_Final["Reco Badges"].str.split("_", n = 2, expand = True)[0]
        GPN_Reco_Final['Sub_Domain'] = GPN_Reco_Final["Reco Badges"].str.split("_", n = 2, expand = True)[1]
        GPN_Reco_Final['Badge'] = GPN_Reco_Final["Reco Badges"].str.split("_", n = 2, expand = True)[2]
        GPN_Reco_Final['key'] = GPN_Reco_Final['Requestor GPN'] + GPN_Reco_Final['Sub_Domain']
        GPN_Reco_Final['rank_temp'] = GPN_Reco_Final.groupby('key')['Prob Score'].rank(ascending=False)
        GPN_Reco_Final = GPN_Reco_Final[GPN_Reco_Final['rank_temp'] == 1]
        # Finalizing output with updated ranks
        GPN_Reco_Final = GPN_Reco_Final.drop(['rank_initial','rank_temp','key','Reco Badges'],axis = 1)
        GPN_Reco_Final['Final_Rank'] = GPN_Reco_Final.groupby('Requestor GPN')['Prob Score'].rank(ascending=False)
        GPN_Reco_5 = GPN_Reco_Final[GPN_Reco_Final['Final_Rank']<6]
        GPN_Reco_5["Key_Badge_SD"] = GPN_Reco_5["Sub_Domain"]+GPN_Reco_5["Badge"]
        df = pd.read_pickle('C:/Users/KPATNAIk/Desktop/data_science/root_data.pkl')
        df = df[df["Initiate a badge date"].notnull()]
        # Find similarity between SLs based on Sub Domain
        #Filter required columns
        badge = df.drop(["Requestor GPN","EmployeeStatusDesc","Badge Status","Badge earned","Initiate a badge date"],axis = 1)
        badge["Time Lapse"] = pd.to_numeric(badge["Time Lapse"])

        badge_final = (badge
                       .groupby(["Key-Badge_SD","Badge Type","Domain","Sub Domain","Badge Classification"], as_index=False)
                       .agg({'Country':pd.Series.nunique,
                                 'Designation':pd.Series.nunique,
                                 'Department':pd.Series.nunique,
                                 'Time Lapse': [np.mean, np.std, pd.Series.nunique, np.min, np.max]})
                      )

        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        # Assigning numerical values and storing in another column
        badge_final['Badge_Type_Cat'] = labelencoder.fit_transform(badge_final['Badge Type'])
        badge_final['Domain_Cat'] = labelencoder.fit_transform(badge_final['Domain'])
        badge_final['Sub_Domain_Cat'] = labelencoder.fit_transform(badge_final['Sub Domain'])
        badge_final['Badge_Classification_Cat'] = labelencoder.fit_transform(badge_final['Badge Classification'])
        # creating one hot encoder object by default
        # entire data passed is one hot encoded
        badge_onehot = badge_final
        enc = OneHotEncoder(handle_unknown='ignore')
        badge_final_Cntry = pd.DataFrame(enc.fit_transform(badge_final[['Badge_Type_Cat']]).toarray())
        badge_final_Cntry.columns = ['Badge_Type_' + str(col) for col in badge_final_Cntry.columns]
        badge_onehot = pd.concat([badge_onehot,badge_final_Cntry], axis=1)
        badge_final_Cntry = pd.DataFrame(enc.fit_transform(badge_final[['Domain_Cat']]).toarray())
        badge_final_Cntry.columns = ['Domian_' + str(col) for col in badge_final_Cntry.columns]
        badge_onehot = pd.concat([badge_onehot,badge_final_Cntry], axis=1)
        badge_final_Cntry = pd.DataFrame(enc.fit_transform(badge_final[['Sub_Domain_Cat']]).toarray())
        badge_final_Cntry.columns = ['Sub_Domain_' + str(col) for col in badge_final_Cntry.columns]
        badge_onehot = pd.concat([badge_onehot,badge_final_Cntry], axis=1)
        badge_final_Cntry = pd.DataFrame(enc.fit_transform(badge_final[['Badge_Classification_Cat']]).toarray())
        badge_final_Cntry.columns = ['Badge_Classification_' + str(col) for col in badge_final_Cntry.columns]
        badge_onehot = pd.concat([badge_onehot,badge_final_Cntry], axis=1)
        badge_onehot = badge_onehot.drop(badge_onehot.columns[1:5], axis =1)
        badge_onehot = badge_onehot[badge_onehot.columns.drop(list(badge_onehot.filter(regex='_Cat')))]
        badge_onehot_numeric = badge_onehot.iloc[:,1:83]
        badge_onehot_numeric = badge_onehot_numeric.apply(pd.to_numeric)
        badge_onehot_numeric[badge_onehot_numeric.columns[4]] = badge_onehot_numeric[badge_onehot_numeric.columns[4]].fillna(0)
        # Normalization
        badge_onehot_normalized = badge_onehot_numeric
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(badge_onehot_normalized)
        badge_onehot_normalized.loc[:,:] = scaled_values

        badge_forsimi = pd.concat([badge_onehot.iloc[:,0],badge_onehot_normalized], axis=1)
        badge_forsimi.set_index(badge_forsimi.iloc[:,0], inplace=True)
        badge_forsimi.drop(badge_forsimi.columns[[0]], axis = 1, inplace = True)
        #Similarity matrix
        # make pairwise distance matrix
        badge_sim = pd.DataFrame(squareform(pdist(badge_forsimi, metric='cosine')),columns = badge_forsimi.index, index = badge_forsimi.index)
        badge_sim_1 = badge_sim.reset_index()
        badge_sim_1.rename(columns={ badge_sim_1.columns[0]: "Key_Badge_SD" }, inplace = True)


        # Transpose using FOR loop (Alter native - Melt function)
        badge_sim_t = pd.concat([badge_sim_1.iloc[:,0],badge_sim_1.iloc[:,1]], axis=1)
        badge_sim_t['Sim with'] = badge_sim_1.columns[1]
        badge_sim_t.rename(columns={ badge_sim_t.columns[1]: "Distance" }, inplace = True)
        col_number = badge_sim_1.shape[1] - 1
        for col in range (2, col_number):
            badge_sim_temp = pd.concat([badge_sim_1.iloc[:,0],badge_sim_1.iloc[:,col]], axis=1)
            badge_sim_temp['Sim with'] = badge_sim_1.columns[col]
            badge_sim_temp.rename(columns={ badge_sim_temp.columns[1]: "Distance" }, inplace = True)
            badge_sim_t = badge_sim_t.append(badge_sim_temp)

        badge_sim_t["Distance"] = 2 - badge_sim_t["Distance"]

        # summing distance to calc relative wt
        badge_sim_wt_grp = pd.pivot_table(badge_sim_t, index = ["Key_Badge_SD"], values = "Distance", aggfunc = [sum])
        # merge total distance with SL table to calculate weightage
        badge_wt_final = pd.merge(badge_sim_t,badge_sim_wt_grp,on = "Key_Badge_SD", how = 'left')
        badge_wt_final.rename(columns={ badge_wt_final.columns[3]: "tot_dist" }, inplace = True)
        badge_wt_final["Relative_Wt"] = badge_wt_final["Distance"]/badge_wt_final["tot_dist"]
        final_badge = pd.merge(GPN_Reco_5,badge_wt_final,on = "Key_Badge_SD", how = 'left')
        final_badge["prob_sort"] = final_badge["Prob Score"]*final_badge["Relative_Wt"]
        final_badge = final_badge.drop_duplicates(['Requestor GPN','Sub_Domain'], keep='first')
        final_badge['Final_Rank'] = final_badge.groupby('Requestor GPN')['prob_sort'].rank(ascending=False)
        final_badge = final_badge[final_badge['Final_Rank']<6]
        final_badge = final_badge.sort_values(by = ["Requestor GPN","Final_Rank"])
        final_badge = final_badge.drop(["Prob Score","Key_Badge_SD","tot_dist","Distance","Sim with","Relative_Wt","prob_sort"],axis = 1)
        data = final_badge
        return render_template('data.html',data=data.to_html())

if __name__ == '__main__':
   app.run("0.0.0.0",threaded=False)
