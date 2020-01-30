# モジュールのimport
from pytrends.request import TrendReq
from itertools import zip_longest
from itertools import filterfalse
import pandas as pd
import csv
import pprint
# 学習用モジュールインポート
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

pytrends = TrendReq(hl='ja-JP', tz=360)
data = pd.read_excel('2016.xlsx')
df = pd.DataFrame(data["キーワードリスト"])
value_list = df.values.tolist()

list3 = []

for list1 in value_list:
    list2 = list1[0].split(',')
    list3.append(list2)

results = []
group_by = 4
for i in range(len(list3)):
    primary_kw = list3[i][0]
    del list3[i][0]
    kw_list = list3[i]
    chunks = zip_longest(*[iter(kw_list)]*group_by)
    p = lambda x: x is None

    merged_df = None
    for elems in list(chunks):
        elems = list(filterfalse(p, elems))
        elems.append(primary_kw)
        pytrends.build_payload(elems, cat=0, timeframe='2016-07-01 2016-07-10', geo='JP', gprop='')
        df = pytrends.interest_over_time()
        # 取得結果のスコアは String になる。 float 変換したいので、True/False　が設定されている`isPartial` を削除して float に変換する。
        del df['isPartial']
        df = df.astype('float64')
        # dataframe　を primary_kw で最大値で正規化する
        scaled_dataframe = df.div(df[primary_kw].max(), axis=0)
        if merged_df is None:
            merged_df = scaled_dataframe
        else:
            # ValueError: columns overlap but no suffix specified が発生するので、やむなく、'postgresql' を削除
            del scaled_dataframe[primary_kw]
            merged_df = merged_df.join(scaled_dataframe)

    results.append(merged_df.mean()/merged_df.mean().mean())

    res = pd.DataFrame(pd.DataFrame(results).sum())
res.columns = [ "trend"]
res["name"] = res.index
res.name = res.name.str.replace("'", "")
res = res.reset_index(drop = True)
res.head()

election = pd.read_csv("2016_cand_ver2.csv")
election.name = election.name.str.replace("　", "")
#election.status = election.where(election.status >= 2, 1)
election = election.drop(["num_votes", "vote_ratio"], axis = 1)
election.sex = (election.sex == "男")

# 党の整理
election.party = election.party.str.replace("おおさか維新の会", "維新").replace("維新政党新風", "維新")
election.party = election.party.str.replace("こころ", "s").replace("支持政党なし",
                              "s").replace("国民怒りの声", "s").replace("新党改革","s").replace("世界経済共同体党",
                              "s").replace("犬丸勝子と共和党", "s").replace("地球平和党", "s").replace("チャレンジド日本",
                              "s").replace("減税日本", "s").replace("S", "s")
pd.unique(election.party)

# 検索トレンドと候補者データのmerge
final = pd.merge(election, res, on = "name", how = "left")
final= final.fillna(0)
final.head()
#　学習用データ（説明変数）の作成
party = pd.get_dummies(final["party"])
party.head()
X = pd.merge(final.drop(['name', 'party','elected', 'area'],axis=1), party, left_index=True, right_index=True)
X.head()
# target
Y = final.elected
# ロジスティック回帰モデルのインスタンスを作成
lr = LogisticRegression()
# ロジスティック回帰モデルの重みを学習
lr.fit(X, Y)
#データの読み込み
data_19 = pd.read_csv("candidates_19.csv")
data_19.head()
#　党の整理
data_19.party = data_19.party.str.replace("立憲", "民進").replace("国民", "民進")
data_19.party = data_19.party.str.replace("安楽", "s").replace("労働者",
                              "s").replace("町田", "s").replace("高橋","s").replace("N国",
                              "s").replace("加藤", "s").replace("オリーブ", "s").replace("れいわ",
                              "s").replace("日本無党派党", "s")
data_19.head()
data = pd.read_excel('2019.xlsx')
df = pd.DataFrame(data["キーワードリスト"])
value_list = df.values.tolist()

list3 = []

for list1 in value_list:
    list2 = list1[0].split(',')
    list3.append(list2)

results = []
group_by = 4
for i in range(len(list3)):
    primary_kw = list3[i][0]
    del list3[i][0]
    kw_list = list3[i]
    chunks = zip_longest(*[iter(kw_list)]*group_by)
    p = lambda x: x is None

    merged_df = None
    for elems in list(chunks):
        elems = list(filterfalse(p, elems))
        elems.append(primary_kw)
        pytrends.build_payload(elems, cat=0, timeframe='2019-07-10 2019-07-19', geo='JP', gprop='')
        df = pytrends.interest_over_time()
        # 取得結果のスコアは String になる。 float 変換したいので、True/False　が設定されている`isPartial` を削除して float に変換する。
        # if 'isPartial' in df.columns:
        if df.empty is not True:
            del df['isPartial']
            df = df.astype('float64')
            # dataframe　を primary_kw で最大値で正規化する
            scaled_dataframe = df.div(df[primary_kw].max(), axis=0)
            if merged_df is None:
                merged_df = scaled_dataframe
            else:
                # ValueError: columns overlap but no suffix specified が発生するので、やむなく、'postgresql' を削除
                del scaled_dataframe[primary_kw]
                merged_df = merged_df.join(scaled_dataframe)
        else:
            print("---------------")
            print("No Data: ", elems)
            print("---------------")

    if merged_df is not None:
        results.append(merged_df.mean()/merged_df.mean().mean())

    #　データ整理
res_19 = pd.DataFrame(pd.DataFrame(results).sum())
res_19.columns = [ "trend"]
res_19["name"] = res_19.index
res_19.name = res_19.name.str.replace("'", "")
res_19 = res_19.reset_index(drop = True)
#　候補者データと検索トレンドをmerge
final_19 = pd.merge(data_19, res_19, on = "name", how = "left")
final_19= final_19.fillna(0)
final_19.head()
# 説明変数の作成
party_19 = pd.get_dummies(final_19["party"])
X_19 = pd.merge(final_19.drop(['name', 'party', 'district_J', 'num'],axis=1), party_19, left_index=True, right_index=True)
X_19.sex = (X_19.sex == "m")
X_19.head()
# 学習モデルを用いた当選確率の予測
prob = lr.predict_proba(X_19)[:, 1]
prob
array([0.158742  , 0.50235483, 0.3621762 , 0.20174061, 0.74296335,
       0.70635213, 0.24673567, 0.37815342, 0.20505149, 0.06710085,
       0.75969601, 0.59558288, 0.25242076, 0.75884715, 0.20369219,
       0.20309021, 0.7613876 , 0.29612055, 0.82600095, 0.12008838,
       0.20490012, 0.57170761, 0.24997479, 0.20429549, 0.52593273,
       0.08597857, 0.23601316, 0.28800128, 0.57050237, 0.60183129,
       0.12614082, 0.76884679, 0.20565778, 0.18877499, 0.04820443,
       0.78260742, 0.29476714, 0.06745056, 0.37684378, 0.33056393,
       0.85009167, 0.55467489, 0.20203994, 0.67937401, 0.81977391,
       0.15074872, 0.06827313, 0.79655594, 0.68705021, 0.78649527,
       0.20084461, 0.12580124, 0.25032349, 0.64301333, 0.49173961,
       0.85346702, 0.91175473, 0.21155181, 0.33931208, 0.68340697,
       0.16665054, 0.3751004 , 0.32879559, 0.20736463, 0.15087743,
       0.61411738, 0.06238836, 0.20024893, 0.20278972, 0.31884302,
       0.21926558, 0.19995159, 0.25007928, 0.75680284, 0.75390698,
       0.14733936, 0.04900315, 0.31904478, 0.32045901, 0.20129224,
       0.44880564, 0.20520294, 0.20339103, 0.87084304, 0.14059122,
       0.65884537, 0.20550608, 0.09226068, 0.81332624, 0.20520294,
       0.59625387, 0.75867713, 0.75884715, 0.59625387, 0.33221036,
       0.20474884, 0.76189348, 0.82304443, 0.08715369, 0.20641751,
       0.74784096, 0.20129224, 0.72717291, 0.20520294, 0.82331483,
       0.59848791, 0.20459764, 0.75969601, 0.67998069, 0.73661433,
       0.1230194 , 0.20429549, 0.81359819, 0.20189024, 0.73535093,
       0.12382332, 0.20189024, 0.37778191, 0.32472147, 0.25743097,
       0.81825942, 0.37662568, 0.20444653, 0.06803716, 0.53554602,
       0.06721723, 0.28065732, 0.20490012, 0.08510667, 0.76273497,
       0.20776107, 0.80207543, 0.16887818, 0.34771381, 0.20596142,
       0.20369219, 0.55846774, 0.61784396, 0.06745056, 0.29592699,
       0.68078864, 0.8905466 , 0.25921054, 0.20444653, 0.15182207,
       0.397249  , 0.20144161, 0.20459764, 0.81218544, 0.29670169,
       0.7770252 , 0.61937797, 0.33179837, 0.05489366, 0.79883739,
       0.08590561, 0.25605973, 0.79962557, 0.8256001 , 0.08481782,
       0.2065697 , 0.78870245, 0.65087186, 0.20490012, 0.4877603 ,
       0.20174061, 0.34111322, 0.33336919, 0.20129224, 0.35584739,
       0.80135571, 0.83145638, 0.32289109, 0.05065497, 0.59915733,
       0.20189024, 0.76239862, 0.3286032 , 0.32167385, 0.08678491,
       0.76020435, 0.20474884, 0.75178335, 0.13681086, 0.20490012,
       0.37444747, 0.20414454, 0.12392412, 0.20369219, 0.81033657,
       0.06571879, 0.65925995, 0.15039232, 0.81922433, 0.80676984,
       0.65863372, 0.82452759, 0.29708948, 0.20520294, 0.7834533 ,
       0.1090507 , 0.20596142, 0.79471198, 0.28840851, 0.20550608,
       0.78434795, 0.18528275, 0.60004932, 0.32126863, 0.0873016 ,
       0.75663055, 0.80307113, 0.32248508, 0.20084461, 0.32350062])
​
