import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from .data_loader import load_partitions
from .preprocessing import permitted_features,clean_infinities
from .config import MODELS,FIGURES,RESULTS

def run():
    train,_=load_partitions(); metadata=json.loads((RESULTS/"model_metadata.json").read_text(encoding="utf-8"))
    surrogate=metadata.get("model_type")=="autogluon"
    model=joblib.load(MODELS/("baseline.joblib" if surrogate else "final_model.joblib")); X=clean_infinities(permitted_features(train.sample(min(100,len(train)),random_state=42)))
    prep=model.named_steps["preprocess"]; clf=model.named_steps["classifier"]; Xt=prep.transform(X)
    if hasattr(Xt,"toarray"): Xt=Xt.toarray()
    Xt=__import__("numpy").asarray(Xt,dtype=float)
    try:
        import shap
        explainer=shap.TreeExplainer(clf); values=explainer.shap_values(Xt,check_additivity=False)
        if isinstance(values,list): values=values[-1]
        if getattr(values,"ndim",0)==3: values=values[:,:,1]
        names=prep.get_feature_names_out(); shap.summary_plot(values,Xt,feature_names=names,show=False,max_display=20); plt.tight_layout(); plt.savefig(FIGURES/"shap_summary.png",bbox_inches="tight"); plt.close()
        imp=abs(values).mean(axis=0); order=imp.argsort()[-20:]
        plt.figure(figsize=(8,6)); plt.barh(names[order],imp[order]); plt.tight_layout(); plt.savefig(FIGURES/"shap_global_importance.png"); plt.close()
        local={"record_index":0,"predicted_class":int(model.predict(X.iloc[[0]])[0]),"top_contributions":sorted([{"feature":str(names[i]),"shap_value":float(values[0,i])} for i in range(len(names))],key=lambda z:abs(z["shap_value"]),reverse=True)[:10],"contribution_interpretation":"SHAP values are for class 1 (Attack): positive values push this model output toward Attack and negative values toward Normal.","method":"TreeExplainer over transformed features; one training-partition sample; contribution values explain model output, not causation","is_surrogate":surrogate,"explained_model":"Random Forest baseline surrogate; the deployed AutoGluon weighted ensemble is evaluated separately" if surrogate else "deployed model"}
        (RESULTS/"shap_local_explanation.json").write_text(json.dumps(local,indent=2),encoding="utf-8")
    except Exception as e:
        (RESULTS/"shap_limitation.txt").write_text(repr(e),encoding="utf-8"); raise

if __name__=="__main__":run()
