import {test} from 'node:test';
import assert from 'node:assert/strict';
import {writeFile, stat, symlink} from 'node:fs/promises';
import {VoiceNotes, MAX_AUDIO} from './notes.mjs';

const item = (type='file', filename='design.pdf') => ({chatInfo:{type:'direct',contact:{contactId:17}},
  chatItem:{meta:{itemId:42},chatDir:{type:'directRcv'},content:{type:'rcvMsgContent',msgContent:{type,text:'Please inspect this'}},
    file:{fileId:7,fileSize:20,fileName:filename}}});
const complete = record => ({type:'rcvFileComplete',chatItem:{chatInfo:{type:'direct',contact:{contactId:17}},
  chatItem:{meta:{itemId:42},file:{fileId:7,fileSource:{filePath:record.path}}}}});

test('file/image invitations expose metadata without downloading before native authorization',async()=>{
  const events=[],commands=[];const owner=new VoiceNotes(async c=>commands.push(c),async e=>events.push(e));
  assert.equal(await owner.invitation(item()),true);
  assert.equal(await owner.invitation(item('image','photo.jpg')),true);
  assert.equal(events[0].type,'attachment_invitation');assert.equal(events[1].media_type,'image');
  assert.equal(events[0].text,'Please inspect this');assert.equal(commands.length,0);
  const outgoing=item();outgoing.chatItem.chatDir.type='directSnd';assert.equal(await owner.invitation(outgoing),false);
  const group=item();group.chatInfo.type='group';assert.equal(await owner.invitation(group),false);
});

test('authorized file transfer reads the verified private destination and cleans temporary bytes',async()=>{
  const events=[],commands=[];const owner=new VoiceNotes(async c=>commands.push(c),async e=>events.push(e));
  await owner.receive({type:'receive_file',media_type:'file',contact:'17',message_id:'42',file_id:7,filename:'design.pdf',text:'Caption'});
  const record=owner.receiving.get(7);assert.equal((await stat(record.dir)).mode&0o777,0o700);
  await writeFile(record.path,Buffer.from('%PDF-1.7 fixture'),{mode:0o600});
  await owner.event(complete(record));
  assert.equal(events.length,1);assert.equal(events[0].type,'attachment');assert.equal(events[0].text,'Caption');
  assert.equal(Buffer.from(events[0].data,'base64').toString(),'%PDF-1.7 fixture');
  assert.equal(events[0].filename,'design.pdf');assert.equal(owner.receiving.size,0);
  await assert.rejects(stat(record.dir),{code:'ENOENT'});assert.match(commands[0],/^\/freceive 7 encrypt=off /);
});

test('file transfers reject traversal and never follow a symlink or oversized completion',async()=>{
  const events=[];const owner=new VoiceNotes(async()=>{},async e=>events.push(e));
  for(const filename of ['../secret','bad/name','bad\\name','..','bad\nname']) {
    await assert.rejects(owner.receive({type:'receive_file',media_type:'file',contact:'17',message_id:'42',file_id:7,filename}));
  }
  await owner.receive({type:'receive_file',media_type:'file',contact:'17',message_id:'42',file_id:7,filename:'file.txt'});
  let record=owner.receiving.get(7);await symlink('/etc/passwd',record.path);await owner.event(complete(record));
  assert.equal(events.at(-1).type,'attachment_transfer_failed');assert.equal(events.some(e=>e.data),false);
  await owner.receive({type:'receive_file',media_type:'file',contact:'17',message_id:'42',file_id:7,filename:'file.txt'});
  record=owner.receiving.get(7);await writeFile(record.path,Buffer.alloc(MAX_AUDIO+1));await owner.event(complete(record));
  assert.equal(events.at(-1).type,'attachment_transfer_failed');assert.equal(owner.receiving.size,0);
});
