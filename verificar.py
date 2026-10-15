"""Verifica la inferencia exportada frente a las métricas del entrenamiento."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'programas'))
from inferencia import predict,evaluate,reservoir
base=Path(__file__).resolve().parent
for kind,n in [('clasificacion',13),('regresion',8),('series',152)]:
 model=json.loads((base/f'resultados/{kind}_modelo.json').read_text())
 ref=json.loads((base/f'resultados/{kind}_metricas.json').read_text())
 data=np.loadtxt(base/f'resultados/{kind}_prueba.csv',delimiter=',')
 actual=evaluate(model,data)
 for metric in ['accuracy','f1_macro','r2','mae','rmse','recall_defect']:
  if metric in actual:assert np.isclose(actual[metric],ref[metric],rtol=1e-8,atol=1e-8),(kind,metric)
 for invalid in [np.zeros(n-1),np.full(n,float('nan')),np.full(n,float('inf'))]:
  try:predict(model,invalid)
  except ValueError:pass
  else:raise AssertionError('Entrada inválida aceptada')
 for f in (base/'programas').glob('*.ipynb'):
  nb=json.loads(f.read_text());assert nb['nbformat']==4
 print(kind, 'OK: métricas, dimensiones y valores no finitos')
# Cambiar el orden temporal cambia los estados de la red recurrente.
x=np.loadtxt(base/'datos/wafer_train.csv',delimiter=',',max_rows=2)[:,1:]
assert not np.allclose(reservoir(x),reservoir(x[:,::-1]))
print('OK: recurrencia sensible al orden de la señal')
