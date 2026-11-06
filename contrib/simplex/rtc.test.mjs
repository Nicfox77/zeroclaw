import {test} from 'node:test';
import assert from 'node:assert/strict';
import {randomBytes} from 'node:crypto';
import {openRtc} from './rtc.mjs';
import {homedir} from 'node:os';
import {join} from 'node:path';

for (const wrongKey of [false, true]) test(wrongKey ? 'wrong encryption key prevents audible audio' : 'encrypted WebRTC transports decoded audio in both directions', {skip: !process.env.SIMPLEX_RTC_TEST}, async () => {
  const config = {assets: process.env.SIMPLEX_CALL_ASSETS ?? join(homedir(), '.local/share/zeroclaw-simplex'),
    executablePath: process.env.CHROMIUM_PATH};
  const received = [0, 0];
  const peers = [];
  const key = randomBytes(32).toString('base64').replaceAll('+', '-').replaceAll('/', '_');
  try {
    for (let i = 0; i < 2; i++) peers.push(await openRtc({...config,
      onSignal: () => {}, onPcm: encoded => {
        const pcm = Buffer.from(encoded, 'base64');
        for (let j = 0; j < pcm.length; j += 2) if (Math.abs(pcm.readInt16LE(j)) > 100) received[i]++;
      }}));
    assert.equal((await peers[1].command({type: 'capabilities', media: 'audio'})).resp.capabilities.encryption, true);
    const offer = (await peers[0].command({type: 'start', media: 'audio', aesKey: key, iceServers: [], relay: false})).resp;
    assert.equal(offer.type, 'offer', offer.message);
    assert.equal(offer.capabilities.encryption, true);
    const answer = (await peers[1].command({type: 'offer', media: 'audio', aesKey: wrongKey ? randomBytes(32).toString('base64').replaceAll('+', '-').replaceAll('/', '_') : key,
      offer: offer.offer, iceCandidates: offer.iceCandidates, iceServers: [], relay: false})).resp;
    assert.equal(answer.type, 'answer', answer.message);
    assert.equal((await peers[0].command({type: 'answer', answer: answer.answer, iceCandidates: answer.iceCandidates})).resp.type, 'ok');
    const connecting = Date.now() + 5000;
    while ((await Promise.all(peers.map(p => p.diagnostics()))).some(s => s.state !== 'connected') && Date.now() < connecting) await new Promise(resolve => setTimeout(resolve, 50));
    assert.ok((await Promise.all(peers.map(p => p.diagnostics()))).every(s => s.state === 'connected'));
    await new Promise(resolve => setTimeout(resolve, 300));
    const samples = 16000;
    const wav = Buffer.alloc(44 + samples * 2);
    wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
    wav.write('data', 36); wav.writeUInt32LE(samples * 2, 40);
    for (let i = 0; i < samples; i++) wav.writeInt16LE(Math.round(8000 * Math.sin(i * 2 * Math.PI * 440 / 16000)), 44 + i * 2);
    await Promise.all(peers.map(peer => peer.play(wav.toString('base64'))));
    const deadline = Date.now() + (wrongKey ? 2000 : 6000);
    while (received.some(count => count < 4000) && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
    if (wrongKey) {
      assert.deepEqual(received, [0, 0]);
      assert.ok((await Promise.all(peers.map(p => p.diagnostics()))).every(s => s.stats.some(s => s.packetsSent > 0)));
    } else assert.ok(received.every(count => count > 4000), `received audible samples: ${received}; ${JSON.stringify(await Promise.all(peers.map(p => p.diagnostics())))}`);
  } finally {await Promise.all(peers.map(peer => peer.close()));}
});
