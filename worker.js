importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');
let py,ready;
async function init(){
 py=await loadPyodide();await py.loadPackage('numpy');
 const code=await (await fetch('inferencia.py')).text();
 await py.runPythonAsync(code.replace("if __name__=='__main__':", "if False:"));
}
onmessage=async({data})=>{try{
 if(!ready)ready=init();await ready;
 const model=await (await fetch(`${data.kind}_modelo.json`)).text();
 py.globals.set('model_text',model);
 await py.runPythonAsync('model=json.loads(model_text)');
 let result;
 if(data.action==='evaluate'){
  const csv=await (await fetch(`${data.kind}_prueba.csv`)).text();py.globals.set('csv_text',csv);
  result=await py.runPythonAsync("import io\ndata=np.loadtxt(io.StringIO(csv_text),delimiter=',')\njson.dumps(evaluate(model,data))");
 }else{
  py.globals.set('input_text',JSON.stringify(data.values));
  result=await py.runPythonAsync('json.dumps(predict(model,json.loads(input_text)))');
 }
 postMessage({ok:true,result:JSON.parse(result),action:data.action});
 }catch(e){postMessage({ok:false,error:String(e)})}
};
