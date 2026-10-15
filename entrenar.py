"""Tres redes neuronales. Origen: ChatGPT de OpenAI, 15/10/2026.
Uso: python entrenar.py [clasificacion|regresion|series|todos]
Ajuste en validación; conjunto de prueba reservado. Datos en ../datos.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys,json,platform,warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
 confusion_matrix, recall_score, r2_score, mean_absolute_error, mean_squared_error)
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_wine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inferencia import reservoir
BASE=Path(__file__).resolve().parent
OUT=BASE/'resultados';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})

def load(name):
 return np.loadtxt(BASE/name,delimiter=',')

def pack(scaler,mlp,target=None):
 out={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
      'weights':[a.tolist() for a in mlp.coefs_],'biases':[a.tolist() for a in mlp.intercepts_],
      'activation':mlp.activation,'out_activation':mlp.out_activation_}
 if target is not None:out.update(target_mean=float(target.mean_[0]),target_scale=float(target.scale_[0]))
 return out

def classify_metrics(y,p):
 return {'accuracy':float(accuracy_score(y,p)),'f1_macro':float(f1_score(y,p,average='macro')),
  'balanced_accuracy':float(balanced_accuracy_score(y,p)),
  'confusion_matrix':confusion_matrix(y,p).tolist()}

def confusion(y,p,names,path):
 cm=confusion_matrix(y,p);fig,ax=plt.subplots(figsize=(6,4))
 ax.imshow(cm,cmap='Blues');ax.set(xticks=range(len(names)),xticklabels=names,
 yticks=range(len(names)),yticklabels=names,xlabel='Predicción',ylabel='Clase real',title='Evaluación con datos de prueba')
 for i in range(len(names)):
  for j in range(len(names)):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=14)
 fig.tight_layout();fig.savefig(OUT/path);plt.close(fig)

def run(kind):
 trials=[]
 if kind=='clasificacion':
  d=load('wine.csv');X=d[:,1:];y=d[:,0].astype(int)
  dev,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
  train,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=42)
  candidates=[((16,),.001),((32,16),.01),((32,),.1)]
  for h,alpha in candidates:
   m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=h,alpha=alpha,solver='lbfgs',max_iter=1500,random_state=42))
   m.fit(X[train],y[train]);score=f1_score(y[val],m.predict(X[val]),average='macro')
   trials.append({'hidden':list(h),'alpha':alpha,'validation_f1_macro':float(score)})
  best=max(range(len(trials)),key=lambda i:trials[i]['validation_f1_macro']);h,alpha=candidates[best]
  m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=h,alpha=alpha,solver='lbfgs',max_iter=1500,random_state=42));m.fit(X[dev],y[dev]);p=m.predict(X[test])
  result=classify_metrics(y[test],p);result['baseline_accuracy']=float(np.mean(y[test]==np.bincount(y[dev]).argmax()))
  model=pack(m[0],m[1]);names=['Tipo 1','Tipo 2','Tipo 3'];confusion(y[test],p,names,'clasificacion.png')
  raw=X[test];labels=y[test];extras={'feature_names':load_wine().feature_names,'classes':names}
 elif kind=='regresion':
  d=load('energy.csv');X=d[:,:8];y=d[:,8]  # Y1 heating load; no Y2 como entrada
  dev,test=train_test_split(np.arange(len(y)),test_size=.2,random_state=42)
  train,val=train_test_split(dev,test_size=.25,random_state=42)
  candidates=[(32,16),(64,32),(32,)]
  def make(h):return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=h,alpha=.01,solver='lbfgs',max_iter=2500,random_state=42)),transformer=StandardScaler())
  for h in candidates:
   m=make(h);m.fit(X[train],y[train]);pv=m.predict(X[val]);trials.append({'hidden':list(h),'alpha':.01,'validation_mae':float(mean_absolute_error(y[val],pv)),'validation_r2':float(r2_score(y[val],pv))})
  best=min(range(len(trials)),key=lambda i:trials[i]['validation_mae']);m=make(candidates[best]);m.fit(X[dev],y[dev]);p=m.predict(X[test])
  result={'r2':float(r2_score(y[test],p)),'mae':float(mean_absolute_error(y[test],p)),'rmse':float(np.sqrt(mean_squared_error(y[test],p))), 'baseline_mae':float(mean_absolute_error(y[test],np.full(len(test),y[dev].mean())))}
  model=pack(m.regressor_[0],m.regressor_[1],m.transformer_)
  fig,ax=plt.subplots(figsize=(6,4));ax.scatter(y[test],p,s=16,alpha=.65);ax.plot([y[test].min(),y[test].max()],[y[test].min(),y[test].max()],'k--');ax.set(xlabel='Carga real',ylabel='Carga estimada',title='Calefacción en edificios  R² = %.4f'%result['r2']);fig.tight_layout();fig.savefig(OUT/'regresion.png');plt.close(fig)
  raw=X[test];labels=y[test];extras={'feature_names':['Compacidad relativa','Superficie','Área de muros','Área de techo','Altura','Orientación','Área acristalada','Distribución del acristalamiento']}
 else:
  a=load('wafer_train.csv');b=load('wafer_test.csv');raw=b[:,1:];labels=(b[:,0]==-1).astype(int)
  X=reservoir(a[:,1:]);Xt=reservoir(raw);y=(a[:,0]==-1).astype(int)
  train,val=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
  dev=np.arange(len(y));test=np.arange(len(labels));candidates=[(16,),(32,),(32,16)]
  for h in candidates:
   m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=h,alpha=.1,solver='lbfgs',max_iter=1000,random_state=42));m.fit(X[train],y[train]);pv=m.predict(X[val]);trials.append({'hidden':list(h),'alpha':.1,'validation_f1_macro':float(f1_score(y[val],pv,average='macro'))})
  best=max(range(len(trials)),key=lambda i:trials[i]['validation_f1_macro']);m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=candidates[best],alpha=.1,solver='lbfgs',max_iter=1000,random_state=42));m.fit(X,y);p=m.predict(Xt)
  result=classify_metrics(labels,p);result.update(recall_defect=float(recall_score(labels,p)),f1_defect=float(f1_score(labels,p)),baseline_accuracy=float(np.mean(labels==0)))
  model=pack(m[0],m[1]);extras={'classes':['Normal','Defectuosa'],'reservoir':{'units':24,'spectral_radius':.8,'leak':.5,'seed':42,'input_scale':.8,'pooling':'final + media + máximo'}}
  confusion(labels,p,extras['classes'],'series.png')
  fig,ax=plt.subplots(figsize=(6,4))
  for cls,name in enumerate(extras['classes']):ax.plot(raw[np.flatnonzero(labels==cls)[0]],label=name)
  ax.set(xlabel='Paso temporal',ylabel='Valor del sensor',title='Secuencias industriales de 152 mediciones');ax.legend();fig.tight_layout();fig.savefig(OUT/'senales.png');plt.close(fig)
 result.update(kind=kind,split={'train':len(train),'validation':len(val),'refit':len(dev),'test':len(test)},trials=trials,selected=trials[best],seed=42,python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__)
 model.update(kind=kind,**extras)
 # Se guardan pesos JSON, nunca objetos pickle ejecutables.
 (OUT/f'{kind}_modelo.json').write_text(json.dumps(model),encoding='utf8')
 (OUT/f'{kind}_metricas.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 np.savetxt(OUT/f'{kind}_prueba.csv',np.column_stack([labels,raw]),delimiter=',',fmt='%.10g')
 log=f'PROGRAMA {kind.upper()}\n'+json.dumps(result,indent=2,ensure_ascii=False)
 print(log);(OUT/f'{kind}_salida.txt').write_text(log,encoding='utf8')
 return result

if __name__=='__main__':
 choice=sys.argv[1] if len(sys.argv)>1 else 'todos'
 if choice not in ['todos','clasificacion','regresion','series']:raise SystemExit('Opción inválida')
 for kind in (['clasificacion','regresion','series'] if choice=='todos' else [choice]):run(kind)
