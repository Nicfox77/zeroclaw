import {test} from 'node:test';
import assert from 'node:assert/strict';
import {WebSocketServer} from 'ws';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {once} from 'node:events';
import {fileURLToPath} from 'node:url';

test('bridge carries private text, excludes sent/group events, and rejects unencrypted calls', async () => {
  const server = new WebSocketServer({host: '127.0.0.1', port: 0}); await once(server, 'listening');
  const child = spawn(process.execPath, [fileURLToPath(new URL('./bridge.mjs', import.meta.url))], {stdio: ['pipe', 'pipe', 'ignore']});
  const lines = createInterface({input: child.stdout});
  const queue = []; let notify;
  lines.on('line', line => {queue.push(JSON.parse(line)); notify?.();});
  async function next() {
    const timeout = Date.now() + 3000;
    while (!queue.length && Date.now() < timeout) await new Promise(resolve => {notify = resolve; setTimeout(resolve, 50);});
    assert.ok(queue.length, 'bridge response timed out'); return queue.shift();
  }
  const send = value => child.stdin.write(JSON.stringify(value) + '\n');
  try {
    send({type: 'init', id: 'init', websocket_url: `ws://127.0.0.1:${server.address().port}`});
    const [socket] = await once(server, 'connection');
    const commands = [];
    socket.on('message', data => {const message = JSON.parse(data); commands.push(message.cmd);
      socket.send(JSON.stringify({corrId: message.corrId, resp: {type: 'cmdOk'}}));});
    assert.deepEqual(await next(), {type: 'result', id: 'init', ok: true});
    send({type: 'send', id: 'send', contact: '17', text: 'A private reply'});
    assert.equal((await next()).ok, true); assert.ok(commands.some(cmd => cmd.startsWith('/_send @17 json ')));
    const item = {chatInfo: {type: 'direct', contact: {contactId: 17}}, chatItem: {meta: {itemId: 42}, chatDir: {type: 'directRcv'}, content: {type: 'rcvMsgContent', msgContent: {type: 'text', text: 'Hello'}}}};
    socket.send(JSON.stringify({resp: {type: 'newChatItems', chatItems: [item]}}));
    assert.deepEqual(await next(), {type: 'message', contact: '17', id: '42', text: 'Hello'});
    item.chatItem.chatDir.type = 'directSnd';
    socket.send(JSON.stringify({resp: {type: 'newChatItems', chatItems: [item]}}));
    socket.send(JSON.stringify({resp: {type: 'callInvitation', callInvitation: {contact: {contactId: 17}, callType: {media: 'audio', capabilities: {encryption: false}}}}}));
    const deadline = Date.now() + 3000;
    while (!commands.some(c => c === '/_call reject @17') && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 20));
    assert.ok(commands.includes('/_call reject @17')); assert.equal(queue.length, 0);
  } finally {child.stdin.end(); child.kill('SIGTERM'); for (const socket of server.clients) socket.terminate(); await new Promise(resolve => server.close(resolve));}
});
