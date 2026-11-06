import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Utterances} from './audio.mjs';

test('silence, barge-in, sentence boundary, and bounded continuous speech', () => {
  let starts = 0; const clips = [];
  const vad = new Utterances(() => starts++, clip => clips.push(clip));
  const silent = Buffer.alloc(2048); const voice = Buffer.alloc(2048);
  for (let i = 0; i < voice.length; i += 2) voice.writeInt16LE(4000, i);
  for (let i = 0; i < 20; i++) vad.push(silent);
  assert.equal(starts, 0);
  for (let i = 0; i < 6; i++) vad.push(voice);
  assert.equal(starts, 1);
  for (let i = 0; i < 12; i++) vad.push(silent);
  assert.equal(clips.length, 1); assert.equal(clips[0].toString('ascii', 0, 4), 'RIFF');
  for (let i = 0; i < 1000; i++) vad.push(voice);
  assert.ok(clips.length >= 3); assert.ok(clips.every(c => c.length <= 962100));
});
