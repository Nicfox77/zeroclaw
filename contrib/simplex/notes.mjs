// The CLI owns transfer IDs and encryption. This owner keeps only private,
// bounded temporary artifacts, never policy or a second conversation history.
import {mkdtemp, chmod, open, rm, realpath} from 'node:fs/promises';
import {constants} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, extname} from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const execute = promisify(execFile);
export const MAX_AUDIO = 5 * 1024 * 1024;
const extensions = new Set(['.m4a', '.mp4', '.wav', '.mp3', '.ogg', '.oga', '.opus', '.webm', '.flac']);
const positive = n => Number.isSafeInteger(n) && n > 0;
const identity = item => ({contact: String(item?.chatInfo?.contact?.contactId), id: String(item?.chatItem?.meta?.itemId)});
async function duration(path) {
  const {stdout} = await execute('ffprobe', ['-v', 'error', '-protocol_whitelist', 'file,pipe',
    '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path],
  {timeout: 15000, maxBuffer: 16384});
  const seconds = Number(stdout.trim());
  if (!Number.isFinite(seconds) || seconds <= 0) throw Error('simplex_invalid_voice_duration');
  return seconds;
}
export class VoiceNotes {
  constructor(rpc, emit, {ttl = 300000} = {}) {
    this.rpc = rpc; this.emit = emit; this.ttl = ttl;
    this.artifacts = new Set(); this.creating = 0; this.closed = false;
  }
  // Transfer lookups are derived from the one owner set (at most 16 records).
  get receiving() {return new Map([...this.artifacts].filter(r => r.direction === 'receive').map(r => [r.fileId, r]));}
  get sending() {return new Map([...this.artifacts].filter(r => r.direction === 'send').map(r => [r.path, r]));}
  async artifact(properties) {
    if (this.closed || this.artifacts.size + this.creating >= 16) throw Error('simplex_voice_transfer_limit');
    // A reservation exists only while the asynchronous directory creation runs.
    this.creating++;
    let dir;
    try {
      dir = await mkdtemp(join(tmpdir(), 'zeroclaw-simplex-voice-'));
      await chmod(dir, 0o700);
      if (this.closed) throw Error('simplex_voice_closed');
      const record = {dir, ...properties}; this.artifacts.add(record); return record;
    } catch (error) {if (dir) await rm(dir, {recursive: true, force: true}); throw error;}
    finally {this.creating--;}
  }
  async remove(record) {
    clearTimeout(record.timer);
    await rm(record.dir, {recursive: true, force: true});
    this.artifacts.delete(record);
  }
  expire(record) {
    record.timer = setTimeout(async () => {
      if (!this.artifacts.has(record)) return;
      if (record.fileId) await this.rpc(`/fcancel ${record.fileId}`).catch(() => {});
      if (!this.artifacts.has(record)) return;
      await this.remove(record);
      await this.emit({type: 'voice_transfer_failed', contact: record.contact, outgoing: record.direction === 'send'});
    }, this.ttl);
    record.timer.unref();
  }
  async invitation(item) {
    const content = item.chatItem?.content;
    if (item.chatInfo?.type !== 'direct' || item.chatItem?.chatDir?.type !== 'directRcv'
      || content?.type !== 'rcvMsgContent' || content.msgContent?.type !== 'voice') return false;
    const file = item.chatItem.file;
    const {contact, id} = identity(item);
    if (!positive(Number(contact)) || !positive(Number(id)) || !positive(file?.fileId)) return true;
    await this.emit({type: 'voice_invitation', contact, id, file_id: file.fileId,
      bytes: file.fileSize, duration: content.msgContent.duration, filename: file.fileName});
    return true;
  }
  async receive(message) {
    const {contact, message_id: id, file_id: fileId, filename, max_duration: maxDuration} = message;
    const ext = extname(filename ?? '').toLowerCase();
    if (!positive(Number(contact)) || !positive(Number(id)) || !positive(fileId)
      || !extensions.has(ext) || !positive(maxDuration)
      || this.receiving.has(fileId)) throw Error('simplex_invalid_voice_invitation');
    const record = await this.artifact({direction: 'receive', contact, id, fileId, maxDuration});
    record.path = join(record.dir, `input${ext}`);
    this.expire(record);
    try {
      // Transport stays encrypted. Only this private, short-lived STT file is
      // plaintext; no untrusted fileSource path is ever opened by the bridge.
      await this.rpc(`/freceive ${fileId} encrypt=off ${record.path}`);
    } catch (error) {await this.remove(record); throw error;}
  }
  async event(resp) {
    if (resp.type === 'rcvFileComplete') {
      const item = resp.chatItem, file = item?.chatItem?.file;
      const record = this.receiving.get(file?.fileId);
      if (!record) return;
      clearTimeout(record.timer);
      try {
        const current = identity(item);
        if (current.contact !== record.contact || current.id !== record.id || file.fileSource?.cryptoArgs
          || await realpath(file.fileSource?.filePath ?? '') !== record.path) throw Error('simplex_voice_transfer_mismatch');
        const handle = await open(record.path, constants.O_RDONLY | constants.O_NOFOLLOW);
        let audio;
        try {
          const stat = await handle.stat();
          if (!stat.isFile() || stat.size <= 0 || stat.size > MAX_AUDIO) throw Error('simplex_voice_size_limit');
          audio = await handle.readFile();
          if (audio.length > MAX_AUDIO) throw Error('simplex_voice_size_limit');
        } finally {await handle.close();}
        const seconds = await duration(record.path);
        if (seconds > record.maxDuration) throw Error('simplex_voice_duration_limit');
        await this.emit({type: 'voice_note', contact: record.contact, id: record.id,
          audio: audio.toString('base64'), filename: `voice${extname(record.path)}`, duration: seconds});
      } catch {
        await this.emit({type: 'voice_transfer_failed', contact: record.contact, outgoing: record.direction === 'send'});
      } finally {await this.remove(record);}
    } else if (['rcvFileError', 'rcvFileSndCancelled'].includes(resp.type)) {
      const fileId = resp.rcvFileTransfer?.fileId ?? resp.chatItem?.chatItem?.file?.fileId;
      const record = this.receiving.get(fileId);
      if (record) {
        await this.remove(record);
        await this.emit({type: 'voice_transfer_failed', contact: record.contact, outgoing: record.direction === 'send'});
      }
    } else if (['sndFileComplete', 'sndFileCompleteXFTP', 'sndFileError', 'sndFileCancelled'].includes(resp.type)) {
      const item = resp.chatItem ?? resp.chatItem_;
      const path = item?.chatItem?.file?.fileSource?.filePath ?? resp.fileTransferMeta?.filePath;
      const record = this.sending.get(path);
      if (!record) return;
      await this.remove(record);
      if (resp.type === 'sndFileError' || resp.type === 'sndFileCancelled')
        await this.emit({type: 'voice_transfer_failed', contact: record.contact, outgoing: record.direction === 'send'});
    }
  }
  async send(contact, text, encoded) {
    let record;
    try {
      const audio = Buffer.from(encoded, 'base64');
      if (!audio.length || audio.length > MAX_AUDIO) throw Error('simplex_voice_size_limit');
      record = await this.artifact({direction: 'send', contact});
      record.path = join(record.dir, 'reply.m4a');
      const handle = await open(record.path, 'wx', 0o600);
      try {await handle.writeFile(audio);} finally {await handle.close();}
      const seconds = await duration(record.path);
      this.expire(record);
      record.admitted = true;
      const chars = [...text], messages = [];
      for (let i = 0; i < chars.length; i += 12000)
        messages.push({msgContent: {type: 'text', text: chars.slice(i, i + 12000).join('')}});
      messages.push({fileSource: {filePath: record.path}, msgContent: {type: 'voice', text: '', duration: Math.ceil(seconds)}});
      // One CLI batch creates the text and note together. Do not remove the
      // source until the asynchronous encrypted upload reaches a terminal event.
      const response = await this.rpc(`/_send @${contact} json ${JSON.stringify(messages)}`);
      const file = response.chatItems?.find(item => item.chatItem?.file)?.chatItem?.file;
      if (file) record.fileId = file.fileId;
    } catch (error) {
      // Before RPC admission, failure proves neither text nor note was sent.
      const admitted = record?.admitted === true;
      if (record && !admitted) await this.remove(record);
      error.notDelivered = !admitted;
      throw error;
    }
  }
  async close() {
    this.closed = true;
    for (const record of this.artifacts) await this.remove(record);
  }
}
