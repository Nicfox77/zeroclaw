import {test} from 'node:test';
import assert from 'node:assert/strict';
import WebSocket,{WebSocketServer} from 'ws';
import {once} from 'node:events';
import {createServer} from 'node:http';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {createHash,randomBytes} from 'node:crypto';
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const listen=async server=>{server.listen(0,'127.0.0.1');await once(server,'listening');return server.address().port;};

async function fixture(idle, summary = false) {
  const root=await mkdtemp(join(tmpdir(),'zeroclaw-idle-proof-'));
  const requests=[],output=[];let sequence=100,socket,daemon,logs='',modelError;
  const vendor=new WebSocketServer({host:'127.0.0.1',port:0});await once(vendor,'listening');
  vendor.on('connection',ws=>{socket=ws;ws.on('message',data=>{
    const {corrId,cmd}=JSON.parse(data);let resp={type:'cmdOk'};
    if(cmd==='/u')resp={type:'activeUser',user:{userId:1}};
    else if(cmd.startsWith('/_get chats'))resp={type:'apiChats',chats:[]};
    else if(cmd.startsWith('/_send ')){
      const messages=JSON.parse(cmd.split(' json ')[1]);
      const wrappers=messages.map(message=>{
        const itemId=++sequence;output.push({text:message.msgContent.text,live:cmd.includes(' live=on')});
        return {chatInfo:{type:'direct',contact:{contactId:17}},chatItem:{meta:{itemId},chatDir:{type:'directSnd'}}};
      });resp={type:'newChatItems',chatItems:wrappers};
    }else if(cmd.startsWith('/_update ')){
      output.push({text:JSON.parse(cmd.split(' json ')[1]).msgContent.text,live:cmd.includes(' live=on')});resp={type:'chatItemUpdated'};
    }else if(cmd.startsWith('/_delete '))resp={type:'chatItemsDeleted'};
    ws.send(JSON.stringify({corrId,resp}));
  });});
  const model=createServer(async(req,res)=>{try{
    let raw='';for await(const part of req)raw+=part;
    if(req.method!=='POST'){res.end('{}');return;}
    const body=JSON.parse(raw),messages=body.messages??[];
    const last=messages.filter(m=>m.role==='user').at(-1)?.content??'';
    const toolResult=messages.at(-1)?.role==='tool';
    requests.push({last,body});
    assert(body.stream,'proof requires the real streaming provider path');
    if(body.model==='summary-model'){
      const prompt=JSON.stringify(messages);
      assert(prompt.includes('Name: Nova')&&prompt.includes('Preferred name: Alex'),'summary must receive the scoped identity briefing');
      assert(prompt.includes("'I' means that assistant")&&prompt.includes('Alex asked me'),'summary must preserve first-person attribution and known names');
      res.writeHead(200,{'Content-Type':'text/event-stream'});
      const emit=delta=>res.write('data: '+JSON.stringify({choices:[{index:0,delta,finish_reason:null}]})+'\n\n');
      // Inference exceeds the one-second allowance while real reasoning keeps
      // advancing. Metadata/empty frames must not supply that activity.
      for(let n=0;n<5&&!res.destroyed;n++){emit({reasoning_content:'private summary reasoning'});await pause(450);}
      if(!res.destroyed){
        emit({content:'I retained our architecture decisions and the user\'s unresolved questions.'});
        res.write('data: '+JSON.stringify({choices:[{index:0,delta:{},finish_reason:'stop'}]})+'\n\n');
        res.end('data: [DONE]\n\n');
      }
      return;
    }
    if(last.includes('prefill'))await pause(3000);
    res.writeHead(200,{'Content-Type':'text/event-stream'});
    const emit=(delta,finish_reason=null)=>{if(!res.destroyed)res.write('data: '+JSON.stringify({choices:[{index:0,delta,finish_reason}]})+'\n\n');};
    emit({role:'assistant'});
    if(last.includes('stalled')){
      if(last.includes('partial'))emit({content:'Interrupted partial.'});
      // Transport keepalives must not turn a stalled model into "progress".
      while(!res.destroyed){res.write(': ping\n\n');await pause(150);}
      return;
    }
    if(last.includes('arguments')&&!toolResult){
      emit({tool_calls:[{index:0,id:'fixture-args',type:'function',function:{name:'shell',arguments:''}}]});
      for(const argumentsPart of ['{"command":"','sleep',' 0','"}']){
        emit({tool_calls:[{index:0,function:{arguments:argumentsPart}}]});await pause(700);
      }
      emit({},'tool_calls');res.end('data: [DONE]\n\n');return;
    }
    if(last.includes('quiet tool')&&!toolResult){
      emit({tool_calls:[{index:0,id:'fixture-quiet',type:'function',function:{name:'shell',arguments:'{"command":"sleep 3"}'}}]});
      emit({},'tool_calls');res.end('data: [DONE]\n\n');return;
    }
    if(toolResult){
      const result=messages.at(-1)?.content??'';
      assert(!String(result).includes('not allowed')&&!String(result).includes('blocked')&&!String(result).includes('Unknown tool'),String(result));
    }
    if(last.includes('long progress')){
      // Longer than the legacy channel's 30s minimum wall-clock cap.
      for(let n=0;n<65&&!res.destroyed;n++){emit({reasoning_content:'private reasoning '});await pause(500);}
    }else if(last.includes('web progress')){
      for(let n=0;n<10&&!res.destroyed;n++){emit({reasoning_content:'private reasoning '});await pause(400);}
    }
    const text=last.includes('quiet tool')?'Quiet tool completed.':last.includes('arguments')?'Arguments completed.':last.includes('prefill')?'Slow prefill completed.':'Progress kept running.';
    emit({content:text});emit({},'stop');if(!res.destroyed)res.end('data: [DONE]\n\n');
  }catch(error){modelError=error;res.destroy();}});
  const modelPort=await listen(model),reservation=createServer(),port=await listen(reservation);
  await new Promise(r=>reservation.close(r));
  const token=randomBytes(32).toString('hex'),hash=createHash('sha256').update(token).digest('hex');
  if(summary){
    const workspace=join(root,'agents','fixture','workspace');await mkdir(workspace,{recursive:true});
    await writeFile(join(workspace,'IDENTITY.md'),'Name: Nova\n');
    await writeFile(join(workspace,'USER.md'),'Preferred name: Alex\n');
  }
  await writeFile(join(root,'config.toml'),`schema_version = 4
[pacing]
turn_idle_timeout_secs = ${idle}
message_timeout_scale_max = 1
[gateway]
host = "127.0.0.1"
port = ${port}
require_pairing = true
paired_tokens = ["${hash}"]
session_persistence = true
check_updates = false
[channels]
session_persistence = true
session_backend = "sqlite"
message_timeout_secs = 1
show_tool_calls = false
[channels.simplex.fixture]
enabled = true
websocket_url = "ws://127.0.0.1:${vendor.address().port}"
bridge_command = [${JSON.stringify(process.execPath)}, ${JSON.stringify(fileURLToPath(new URL('./bridge.mjs',import.meta.url)))}]
[peer_groups.fixture]
channel = "simplex.fixture"
external_peers = ["17"]
[memory]
backend = "none"
auto_save = false
[providers.models.ollama.fixture]
model = "fixture-model"
uri = "http://127.0.0.1:${modelPort}"
timeout_secs = 30
context_window = 128000
[providers.models.ollama.summary]
model = "summary-model"
uri = "http://127.0.0.1:${modelPort}"
timeout_secs = 30
context_window = 128000
[risk_profiles.fixture]
level = "full"
allowed_commands = ["sleep"]
[runtime_profiles.fixture]
max_context_tokens = 64000
max_history_messages = ${summary ? 6 : 200}
max_tool_iterations = 5
shell_timeout_secs = 10
[runtime_profiles.fixture.summary_deck]
enabled = ${summary}
timeout_secs = 1
max_summary_tokens = 256
[agents.fixture]
model_provider = "ollama.fixture"
summary_provider = "ollama.summary"
risk_profile = "fixture"
runtime_profile = "fixture"
channels = ["simplex.fixture"]
[agents.fixture.main_session]
enabled = true
web = true
[agents.fixture.main_session.simplex]
fixture = "17"
`,{mode:0o600});
  async function until(predicate,label,timeout=45000){const deadline=Date.now()+timeout;while(Date.now()<deadline){if(modelError)throw modelError;if(daemon?.exitCode!==null&&daemon?.exitCode!==undefined)assert.fail('fixture daemon exited: '+logs);const value=predicate();if(value)return value;await pause(40);}assert.fail(label+'; output='+JSON.stringify(output)+'; logs='+logs.slice(-5000));}
  daemon=spawn(process.env.ZEROCLAW_TEST_BINARY??fileURLToPath(new URL('../../target/release/zeroclaw',import.meta.url)),['--config-dir',root,'daemon'],{
    env:{...process.env,ZEROCLAW_CONFIG_DIR:root},stdio:['ignore','pipe','pipe'],
  });
  for(const stream of [daemon.stdout,daemon.stderr])stream.on('data',data=>{logs=(logs+data).slice(-18000);});
  async function close(){
    const stopped=once(daemon,'exit');daemon.kill('SIGTERM');await Promise.race([stopped,pause(5000)]);
    if(daemon.exitCode===null){daemon.kill('SIGKILL');await stopped;}
    for(const client of vendor.clients)client.terminate();await new Promise(r=>vendor.close(r));
    model.closeAllConnections();await new Promise(r=>model.close(r));await rm(root,{recursive:true,force:true});
  }
  const send=text=>socket.send(JSON.stringify({resp:{type:'newChatItems',chatItems:[{chatInfo:{type:'direct',contact:{contactId:17}},
    chatItem:{meta:{itemId:++sequence},chatDir:{type:'directRcv'},content:{type:'rcvMsgContent',msgContent:{type:'text',text}}},
  }]}}));
  async function web(text){
    const ws=new WebSocket(`ws://127.0.0.1:${port}/ws/chat?agent=fixture&session_id=idle-proof&token=${token}`);
    const frames=[];ws.on('message',data=>frames.push(JSON.parse(data)));
    try{
      await once(ws,'open');await until(()=>frames.find(f=>f.type==='session_start'),'web session');
      assert.equal(frames.find(f=>f.type==='session_start').session_id,'main-session-fixture');
      ws.send(JSON.stringify({type:'message',content:text}));
      return await until(()=>frames.find(f=>f.type==='done'||f.type==='error'),'web terminal response');
    }finally{ws.terminate();}
  }
  try{await until(()=>socket,'bridge ready');}
  catch(error){await close();throw error;}
  return {requests,output,until,send,web,close,logs:()=>logs};
}

test('summary deck honors the agent summarizer and permits inference longer than its idle allowance',{
  skip:process.env.SIMPLEX_DAEMON_TEST!=='1',timeout:90000,
},async()=>{
  const f=await fixture(10,true);
  try{
    for(let n=0;n<9;n++){
      const result=await f.web('fixture web quick turn '+n);
      assert.equal(result.type,'done',JSON.stringify(result));
    }
    assert(f.requests.some(r=>r.body.model==='summary-model'),'configured summary provider must receive the job');
    assert(f.requests.some(r=>r.body.model==='fixture-model'&&JSON.stringify(r.body.messages).includes('I retained our architecture decisions')),'published summary must reach a later real chat request');
    assert(!f.logs().includes('summary_deck_timeout'),'actual progress must keep summary inference alive');
  }finally{await f.close();}
});

test('activity mode survives the old wall clock, quiet tools and argument generation; real stalls stop on SimpleX and web',{
  skip:process.env.SIMPLEX_DAEMON_TEST!=='1',timeout:180000,
},async()=>{
  const f=await fixture(2);
  try{
    const first=Date.now();f.send('fixture long progress');
    await f.until(()=>f.output.find(i=>i.text==='Progress kept running.'&&!i.live),'long progressing turn');
    assert(Date.now()-first>30000,'must outlive the legacy minimum total cap');
    f.send('fixture quiet tool');
    const quiet=Date.now();await f.until(()=>f.output.find(i=>i.text==='Quiet tool completed.'&&!i.live),'quiet bounded shell');
    assert(Date.now()-quiet>2500,'the quiet shell must actually execute');
    f.send('fixture arguments');await f.until(()=>f.output.find(i=>i.text==='Arguments completed.'&&!i.live),'incremental tool arguments');
    f.send('fixture stalled partial');
    await f.until(()=>f.output.find(i=>i.text.includes('without model or tool progress')),'SimpleX idle notice');
    assert(f.output.some(i=>i.text.includes('Interrupted partial.')&&i.text.includes('[Interrupted]')));
    assert.equal(f.requests.filter(r=>String(r.last).includes('fixture stalled partial')).length,1,'stalled output must never be replayed');
    const progress=await f.web('fixture web progress');
    assert.equal(progress.type,'done');assert(progress.full_response.includes('Progress kept running.'));
    const stalled=await f.web('fixture web stalled');
    assert.equal(stalled.type,'error');assert(JSON.stringify(stalled).includes('without model or tool progress'));
    const recovered=await f.web('fixture web progress');assert.equal(recovered.type,'done','a timeout cannot poison subsequent turns');
    assert(!f.logs().includes('session_capture_stopped'),'ordinary model inactivity must leave capture usable');
  }finally{await f.close();}
});

test('a larger configurable idle window permits slow uncached model prefill',{
  skip:process.env.SIMPLEX_DAEMON_TEST!=='1',timeout:60000,
},async()=>{
  const f=await fixture(5);
  try{
    const started=Date.now(),response=await f.web('fixture slow prefill');
    assert.equal(response.type,'done');assert(response.full_response.includes('Slow prefill completed.'));
    assert(Date.now()-started>=2900);
  }finally{await f.close();}
});
