import {test} from 'node:test';
import assert from 'node:assert/strict';
import {writeFile, readFile, stat, symlink} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {VoiceNotes, MAX_AUDIO} from './notes.mjs';

const makeAudio = () => {
  const result = spawnSync('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
    '-t', '0.3', '-c:a', 'aac', '-movflags', 'frag_keyframe+empty_moov', '-f', 'mp4', 'pipe:1']);
  assert.equal(result.status, 0); return result.stdout;
};
const item = (fileId = 7) => ({chatInfo: {type: 'direct', contact: {contactId: 17}},
  chatItem: {meta: {itemId: 42}, chatDir: {type: 'directRcv'}, content: {type: 'rcvMsgContent',
    msgContent: {type: 'voice', text: '', duration: 1}}, file: {fileId, fileName: 'voice.m4a', fileSize: 2000}}});
const receive = {contact: '17', message_id: '42', file_id: 7, filename: 'voice.m4a', max_duration: 600};

test('voice invitations require native admission before download; receive only opens its private owned file', async () => {
  const commands = [], events = [];
  const notes = new VoiceNotes(async cmd => {commands.push(cmd);}, async value => events.push(value));
  try {
    assert.equal(await notes.invitation(item()), true);
    assert.equal(commands.length, 0);
    assert.equal(events[0].type, 'voice_invitation');
    await notes.receive(receive);
    const path = commands[0].split(' encrypt=off ')[1];
    assert.ok(path.endsWith('/input.m4a'));
    assert.equal((await stat(notes.receiving.get(7).dir)).mode & 0o777, 0o700);
    const audio = makeAudio(); await writeFile(path, audio, {mode: 0o600});
    const complete = item(); complete.chatItem.file.fileSource = {filePath: path};
    await notes.event({type: 'rcvFileComplete', chatItem: complete});
    assert.equal(events[1].type, 'voice_note');
    assert.equal(events[1].id, '42'); assert.equal(events[1].contact, '17');
    assert.deepEqual(Buffer.from(events[1].audio, 'base64'), audio);
    assert.ok(events[1].duration > 0 && events[1].duration < 1);
    await assert.rejects(stat(path), {code: 'ENOENT'});
  } finally {await notes.close();}
});

test('unowned files, contact mismatch, oversized files, and symlinks never reach STT', async () => {
  for (const attack of ['path', 'contact', 'size', 'symlink']) {
    const events = []; const notes = new VoiceNotes(async () => {}, async value => events.push(value));
    try {
      await notes.receive(receive);
      const path = notes.receiving.get(7).path;
      if (attack === 'symlink') await symlink('/etc/passwd', path);
      else await writeFile(path, attack === 'size' ? Buffer.alloc(MAX_AUDIO + 1) : makeAudio());
      const complete = item(); complete.chatItem.file.fileSource = {filePath: attack === 'path' ? '/etc/passwd' : path};
      if (attack === 'contact') complete.chatInfo.contact.contactId = 18;
      await notes.event({type: 'rcvFileComplete', chatItem: complete});
      assert.deepEqual(events.map(e => e.type), ['voice_transfer_failed']);
      assert.equal(notes.receiving.size, 0);
      await assert.rejects(stat(path), {code: 'ENOENT'});
    } finally {await notes.close();}
  }
});

test('one batch creates text plus a native voice note; source survives ACK until encrypted upload completes', async () => {
  const commands = [], events = [];
  const notes = new VoiceNotes(async cmd => {commands.push(cmd); return {type: 'newChatItems', chatItems: []};}, async value => events.push(value));
  try {
    const audio = makeAudio(); await notes.send('17', 'The final reply.', audio.toString('base64'));
    assert.equal(commands.length, 1);
    const batch = JSON.parse(commands[0].slice('/_send @17 json '.length));
    assert.equal(batch[0].msgContent.text, 'The final reply.');
    assert.equal(batch[1].msgContent.type, 'voice'); assert.equal(batch[1].msgContent.duration, 1);
    const path = batch[1].fileSource.filePath;
    assert.deepEqual(await readFile(path), audio);
    assert.equal((await stat(path)).mode & 0o777, 0o600);
    const complete = item(); complete.chatItem.file.fileSource = {filePath: path};
    await notes.event({type: 'sndFileCompleteXFTP', chatItem: complete});
    await assert.rejects(stat(path), {code: 'ENOENT'}); assert.equal(notes.sending.size, 0);
    assert.equal(events.length, 0);
  } finally {await notes.close();}
});

test('failed probe is definitely unsent; lost CLI acceptance stays uncertain; shutdown removes artifacts', async () => {
  const notes = new VoiceNotes(async () => {throw Error('lost_acceptance');}, async () => {});
  try {
    await assert.rejects(notes.send('17', 'reply', Buffer.from('invalid audio').toString('base64')), error => error.notDelivered === true);
    await assert.rejects(notes.send('17', 'reply', makeAudio().toString('base64')), error => error.notDelivered === false);
    const paths = [...notes.sending.keys()]; assert.equal(paths.length, 1);
    await notes.close();
    for (const path of paths) await assert.rejects(stat(path), {code: 'ENOENT'});
  } finally {await notes.close();}
});

test('timeouts cancel admitted transfers, notify, and remove temporary files', async () => {
  const events = [], commands = [];
  const notes = new VoiceNotes(async cmd => commands.push(cmd), async value => events.push(value), {ttl: 25});
  try {
    await notes.receive(receive); const path = notes.receiving.get(7).path;
    await new Promise(resolve => setTimeout(resolve, 100));
    assert.ok(commands.includes('/fcancel 7'));
    assert.equal(notes.receiving.size, 0);
    assert.equal(events[0].type, 'voice_transfer_failed');
    await assert.rejects(stat(path), {code: 'ENOENT'});
  } finally {await notes.close();}
});
