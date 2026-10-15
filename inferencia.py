"""Inferencia Python con NumPy. Misma red entrenada por entrenar.py.
En navegador se ejecuta con Pyodide; no hay predicciones pregrabadas.
"""
import json
from pathlib import Path
import numpy as np

def reservoir(X):
 """Red recurrente Echo State de 24 unidades; procesa 152 pasos en orden."""
 X=np.asarray(X,dtype=float);n=24;rng=np.random.default_rng(42)
 wi=rng.normal(0,.8,(1,n));w=rng.normal(0,1,(n,n))
 w*=.8/np.max(np.abs(np.linalg.eigvals(w)));bias=rng.uniform(-.2,.2,n)
 h=np.zeros((len(X),n));s=np.zeros_like(h);mx=np.full_like(h,-np.inf)
 for t in range(X.shape[1]):
  h=.5*h+.5*np.tanh(X[:,t:t+1]@wi+h@w+bias)
  s+=h;mx=np.maximum(mx,h)
 return np.hstack([h,s/X.shape[1],mx])

def predict(model,X):
 X=np.asarray(X,dtype=float)
 if X.ndim==1:X=X[None,:]
 expected={'clasificacion':13,'regresion':8,'series':152}[model['kind']]
 if X.ndim!=2 or X.shape[1]!=expected or not np.isfinite(X).all():
  raise ValueError(f'Se requieren {expected} números finitos por muestra')
 H=reservoir(X) if model['kind']=='series' else X
 H=(H-np.array(model['mean']))/np.array(model['scale'])
 for i,(w,b) in enumerate(zip(model['weights'],model['biases'])):
  H=H@np.array(w)+np.array(b)
  if i<len(model['weights'])-1:H=np.maximum(H,0)
 if model['out_activation']=='identity':return (H[:,0]*model['target_scale']+model['target_mean']).tolist()
 if model['out_activation']=='logistic':
  p=1/(1+np.exp(-np.clip(H[:,0],-700,700)));return np.column_stack([1-p,p]).tolist()
 H=np.exp(H-H.max(axis=1,keepdims=True));return (H/H.sum(axis=1,keepdims=True)).tolist()

def evaluate(model,data):
 data=np.asarray(data,dtype=float);y=data[:,0];p=np.array(predict(model,data[:,1:]))
 if model['kind']=='regresion':
  return {'n':len(y),'r2':float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)),
          'mae':float(np.mean(abs(y-p))),'rmse':float(np.sqrt(np.mean((y-p)**2)))}
 pred=p.argmax(axis=1);k=p.shape[1];cm=np.zeros((k,k),int)
 for a,b in zip(y.astype(int),pred):cm[a,b]+=1
 recalls=np.diag(cm)/cm.sum(axis=1);f1=2*np.diag(cm)/(cm.sum(axis=0)+cm.sum(axis=1))
 r={'n':len(y),'accuracy':float(np.mean(y==pred)),'f1_macro':float(f1.mean()),'balanced_accuracy':float(recalls.mean()),'confusion_matrix':cm.tolist()}
 if model['kind']=='series':r.update(recall_defect=float(recalls[1]),f1_defect=float(f1[1]))
 return r

if __name__=='__main__':
 import sys
 kind=sys.argv[1] if len(sys.argv)>1 else 'clasificacion'
 base=Path(__file__).resolve().parent
 if (base/'resultados'/f'{kind}_modelo.json').exists():base=base/'resultados'
 model=json.loads((base/f'{kind}_modelo.json').read_text())
 data=np.loadtxt(base/f'{kind}_prueba.csv',delimiter=',')
 print(json.dumps(evaluate(model,data),indent=2))
