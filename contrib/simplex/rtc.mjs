// SimpleX's separately installed, unmodified WebRTC assets own encryption and
// SDP compatibility. This module supplies virtual audio, never a physical mic.
import {chromium} from 'playwright-core';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';

export async function openRtc({assets, executablePath, onSignal, onPcm}) {
  const browser = await chromium.launch({executablePath, headless: true,
    args: ['--autoplay-policy=no-user-gesture-required']});
  try {
    const page = await browser.newPage();
    // No HTTP listener and no external scripts. Media is virtual; getUserMedia
    // is replaced before the upstream implementation can request permissions.
    await page.route('**/*', route => route.request().url() === 'http://127.0.0.1/simplex-call'
      ? route.fulfill({contentType: 'text/html', body: '<video id="local-video-stream" muted></video><video id="local-screen-video-stream" muted></video><video id="remote-video-stream" muted></video><video id="remote-screen-video-stream" muted></video>'})
      : route.abort());
    await page.goto('http://127.0.0.1/simplex-call');
    await page.exposeBinding('hostSignal', (_source, signal) => onSignal(signal));
    await page.exposeBinding('hostPcm', (_source, pcm) => onPcm(pcm));
    await page.evaluate(() => {
      const audio = new AudioContext({sampleRate: 16000});
      const destination = audio.createMediaStreamDestination();
      const silence = audio.createConstantSource();
      silence.offset.value = 0;
      silence.connect(destination);
      silence.start();
      const sources = new Set();
      const waiting = new Set();
      let next = 0;
      globalThis.playAudio = async base64 => {
        const raw = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
        const buffer = await audio.decodeAudioData(raw.buffer);
        if (buffer.duration > 180 || sources.size >= 2) throw Error('simplex_audio_queue_limit');
        const source = audio.createBufferSource();
        source.buffer = buffer;
        source.connect(destination);
        next = Math.max(next, audio.currentTime + 0.025);
        const starts = next;
        source.start(starts);
        next += buffer.duration;
        sources.add(source);
        source.onended = () => sources.delete(source);
        // ACK when playback begins. The producer can synthesize the next clip
        // while this one plays, with at most one future clip queued.
        await new Promise((resolve, reject) => {
          const cancel = () => {clearTimeout(timer); reject(Error('simplex_audio_cancelled'));};
          const timer = setTimeout(() => {waiting.delete(cancel); resolve();}, Math.max(0, starts - audio.currentTime) * 1000);
          waiting.add(cancel);
        });
      };
      globalThis.stopAudio = () => {
        for (const source of sources) source.stop();
        for (const cancel of waiting) cancel();
        waiting.clear();
        sources.clear(); next = 0;
      };
      Object.defineProperty(navigator, 'mediaDevices', {value: {
        getUserMedia: async () => destination.stream,
        enumerateDevices: async () => [{kind: 'audioinput', deviceId: 'virtual'}],
      }});
      const Original = RTCPeerConnection;
      globalThis.RTCPeerConnection = class extends Original {
        constructor(...args) {
          super(...args);
          this.addEventListener('track', event => {
            if (event.track.kind !== 'audio') return;
            const input = audio.createMediaStreamSource(new MediaStream([event.track]));
            const processor = audio.createScriptProcessor(1024, 1, 1);
            input.connect(processor); processor.connect(audio.destination);
            let pending = false;
            processor.onaudioprocess = event => {
              if (pending) return; // bounded host IPC; never queue raw audio
              const data = event.inputBuffer.getChannelData(0);
              const bytes = new Uint8Array(data.length * 2);
              const view = new DataView(bytes.buffer);
              for (let i = 0; i < data.length; i++) view.setInt16(i * 2, Math.max(-32768, Math.min(32767, data[i] * 32767)), true);
              pending = true;
              hostPcm(btoa(String.fromCharCode(...bytes))).finally(() => {pending = false;});
            };
          });
        }
      };
      audio.resume();
      globalThis.audioState = () => audio.state;
    });
    for (const file of ['lz-string.min.js', 'call.js']) {
      await page.addScriptTag({content: await readFile(join(assets, file), 'utf8')});
    }
    await page.evaluate(() => {sendMessageToNative = signal => hostSignal(signal);});
    return {
      command: command => page.evaluate(command => processCommand({command}), command),
      play: audio => page.evaluate(audio => playAudio(audio), audio),
      stop: () => page.evaluate(() => stopAudio()),
      diagnostics: () => page.evaluate(async () => ({audio: audioState(),
        state: activeCall?.connection.connectionState,
        local: activeCall?.localStream.getAudioTracks().map(t => ({enabled: t.enabled, state: t.readyState})),
        stats: [...await activeCall.connection.getStats()].map(([,s]) => s).filter(s => ['inbound-rtp', 'outbound-rtp'].includes(s.type) && s.kind === 'audio').map(s => ({type: s.type, packetsReceived: s.packetsReceived, packetsSent: s.packetsSent, audioLevel: s.audioLevel}))})),
      close: () => browser.close(),
    };
  } catch (error) {await browser.close(); throw error;}
}
