import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from .data_loader import load_partitions
from .preprocessing import permitted_features, build_preprocessor, clean_infinities
from .config import RESULTS, SEED
from sklearn.pipeline import Pipeline

def run():
    train,_=load_partitions(); data=train.sample(min(30000,len(train)),random_state=SEED)
    y=data.label.astype(int); X=clean_infinities(permitted_features(data)); Xa,Xv,ya,yv=train_test_split(X,y,test_size=.25,random_state=SEED,stratify=y)
    out=[]
    for name,features in (("valid",Xa),("intentional_target_copy",Xa.assign(__intentional_leak_label=ya))):
        val=Xv if name=="valid" else Xv.assign(__intentional_leak_label=yv)
        pp=build_preprocessor(features); model=Pipeline([("preprocess",pp),("classifier",RandomForestClassifier(n_estimators=100,min_samples_leaf=2,n_jobs=2,random_state=SEED))]); model.fit(features,ya); pred=model.predict(val)
        out.append({"experiment":name,"accuracy":accuracy_score(yv,pred),"f1":f1_score(yv,pred),"valid_for_deployment":name=="valid"})
    pd.DataFrame(out).to_csv(RESULTS/"leakage_results.csv",index=False)
    (RESULTS/"leakage_explanation.md").write_text("The second row deliberately adds an exact copy of the training label as `__intentional_leak_label`. It is a synthetic diagnostic and is excluded from all valid model comparisons and deployment. A near-perfect score demonstrates target exposure, not traffic detection ability. The supplied `id` was separately excluded because it is a partition-local sequence identifier.\n",encoding="utf-8")

if __name__=="__main__":run()
