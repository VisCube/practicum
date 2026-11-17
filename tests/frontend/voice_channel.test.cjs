const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('src/assistant/api/static/index.html', 'utf8');

function section(start, end) {
  const a = html.indexOf(start);
  const b = html.indexOf(end, a);
  assert.ok(a >= 0 && b > a);
  return html.slice(a, b);
}

test('disconnect discards partial utterance and stale socket cannot affect new connection', async () => {
  let aborts = 0;
  let reconnects = 0;
  class FakeWebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 1; }
    send() {}
    close() {}
  }
  const context = vm.createContext({
    WebSocket: FakeWebSocket, console, setTimeout, clearTimeout, setInterval, clearInterval,
    voice: { ws: null, ready: false, pingTimer: null },
    app: { voiceOut: true, ending: false }, UI_BUILD: 'test',
    WS_OPEN_TIMEOUT_MS: 5000, WS_READY_TIMEOUT_MS: 10000, WS_HEARTBEAT_INTERVAL_MS: 15000,
    heartbeatTick: () => {},
    voiceUrl: () => 'ws://example/ws/call',
    voiceSend: () => true, setChannelLabel: () => {}, refreshLiveView: () => {},
    updateMicButton: () => {}, handleVoiceMessage: () => {}, inCall: () => true, showStatus: () => {},
    abortUtterance: () => { aborts++; }, reconnectVoice: () => { reconnects++; },
  });
  vm.runInContext(section('    function openWsChannel(', '    function handleVoiceMessage('), context);
  const opening = context.openWsChannel('123');
  const ws = context.voice.ws;
  ws.onopen();
  ws.onmessage({ data: JSON.stringify({ t: 'ready', call_id: '1' }) });
  await opening;
  assert.equal(context.voice.ready, true);
  ws.onclose({ code: 1006, reason: '' });
  assert.equal(context.voice.ready, false);
  assert.equal(aborts, 1);
  assert.equal(reconnects, 1);
  const next = { readyState: 1 };
  context.voice.ws = next;
  context.voice.ready = true;
  ws.onclose({ code: 1006, reason: '' });
  assert.equal(context.voice.ws, next);
  assert.equal(context.voice.ready, true);
  assert.equal(aborts, 1);
});

test('audio frames are not buffered or sent while the channel is not ready', () => {
  let converted = 0;
  const context = vm.createContext({
    micListening: () => false,
    floatToInt16: () => { converted++; },
  });
  vm.runInContext(section('    function onAudioFrame(', '    /* =========================================================\n       Композер'), context);
  context.onAudioFrame(new Float32Array(32));
  assert.equal(converted, 0);
});

test('capture opens even while Bluetooth audio context is temporarily suspended', async () => {
  const nodes = [];
  const sizes = [];
  const ctx = {
    state: 'suspended', sampleRate: 8000, destination: {},
    createGain: () => ({ gain: {}, connect() {}, disconnect() {} }),
    createMediaStreamSource: () => ({ connect() {}, disconnect() {} }),
    createScriptProcessor: (size) => {
      sizes.push(size);
      const node = { connect() {}, disconnect() {} };
      nodes.push(node);
      return node;
    },
  };
  const track = { muted: true, readyState: 'live', addEventListener() {} };
  const stream = { getAudioTracks: () => [track] };
  const context = vm.createContext({
    app: { mode: 'asr' }, performance,
    capture: { processor: null, openingPromise: null, ctx: null }, captureEpoch: 0,
    micStream: { stream: null }, ctx, stream,
    micTracksLive: s => s === stream, audioContextUsable: () => true,
    disconnectCaptureNodes: () => {}, ensureAudioContext: async () => ctx,
    acquireMic: async () => stream, tryResumeAudioContext: async () => {},
    captureWanted: () => true, stopCaptureIfIdle: () => {},
    standMic: { keep: false }, renderStandMic: () => {},
    createSharedMeter: () => ({}), updateAsrUi: () => {}, audioDebug: () => {},
    createCaptureProcessor: async (c, size) => ({ node: c.createScriptProcessor(size), kind: 'script' }),
    setTimeout: () => 0, Date, console,
  });
  vm.runInContext(section('    async function ensureCaptureOpen()', '    async function openCallMic(opts)'), context);
  const result = await context.ensureCaptureOpen();
  assert.equal(result.ctx, ctx);
  assert.equal(result.coldOpen, true);
  assert.equal(context.capture.processor, nodes[0]);
  assert.deepEqual(sizes, [512]);
  context.micStream.stream = stream;
  const again = await context.ensureCaptureOpen();
  assert.equal(again.coldOpen, false);
  assert.equal(nodes.length, 1);
  context.capture.processor = null;
  context.micStream.stream = null;
  ctx.sampleRate = 48000;
  await context.ensureCaptureOpen();
  assert.deepEqual(sizes, [512, 2048]);
});

test('cancelled pending microphone request does not block reopening or replace new stream', async () => {
  let resolveOld;
  let resolveNew;
  let opens = 0;
  let stopped = false;
  const oldTrack = { readyState: 'live', stop: () => { stopped = true; } };
  const newTrack = { readyState: 'live', label: 'test', getSettings: () => ({}), addEventListener() {} };
  const oldStream = { getAudioTracks: () => [oldTrack], getTracks: () => [oldTrack] };
  const newStream = { getAudioTracks: () => [newTrack], getTracks: () => [newTrack] };
  const context = vm.createContext({
    captureEpoch: 0, capture: { openingPromise: null },
    micStream: { stream: null, streamPromise: null }, mic: {}, app: {},
    setTimeout, clearTimeout, console, audioDebug: () => {},
    micGetUserMediaWithRetry: () => new Promise(resolve => {
      opens++;
      if (opens === 1) resolveOld = resolve;
      else resolveNew = resolve;
    }),
    micTracksLive: stream => Boolean(stream), renderMicHint: () => {},
    disconnectCaptureNodes: () => {}, releaseMic: () => {},
    updateMicButton: () => {}, updateAsrUi: () => {},
  });
  vm.runInContext(section('    async function acquireMic()', '    function releaseMic()'), context);
  vm.runInContext(section('    function teardownCapture()', '    function stopCaptureIfIdle()'), context);
  const first = context.acquireMic();
  context.teardownCapture();
  const second = context.acquireMic();
  assert.equal(opens, 2);
  resolveNew(newStream);
  assert.equal(await second, newStream);
  resolveOld(oldStream);
  await assert.rejects(first, /захват отменён/);
  assert.equal(stopped, true);
  assert.equal(context.micStream.stream, newStream);
});

test('microphone processing is optional and is enabled by default', () => {
  let stored = 'standard';
  const context = vm.createContext({ localStorage: { getItem: () => stored } });
  vm.runInContext(section('    function micAudioConstraints()', '    async function requestMicPermission()'), context);
  let settings = context.micAudioConstraints();
  assert.equal(settings.echoCancellation, true);
  assert.equal(settings.noiseSuppression, true);
  assert.equal(settings.autoGainControl, true);
  stored = 'raw';
  settings = context.micAudioConstraints();
  assert.equal(settings.channelCount, 1);
  assert.equal(settings.echoCancellation, false);
  assert.equal(settings.noiseSuppression, false);
  assert.equal(settings.autoGainControl, false);
});

test('TTS playback errors are not mistaken for a successful interruption', async () => {
  for (const name of ['AbortError', 'NotSupportedError', 'NotAllowedError']) {
    const context = vm.createContext({
      speechSeq: 1, speaker: {}, console,
      URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} },
      Audio: class { play() { return Promise.reject({ name, message: 'test' }); } removeAttribute() {} load() {} remove() {} },
      document: { body: { appendChild() {} } },
      mediaErrorText: () => '',
    });
    vm.runInContext(section('    function playViaElement(', '    async function playServerAudio('), context);
    const result = await context.playViaElement({}, 'audio/wav');
    assert.equal(result.ok, false, name);
    assert.match(result.reason, new RegExp(name));
    assert.equal(context.speaker.finish, null);
  }
});

test('an explicitly cancelled playback remains interrupted', async () => {
  let rejectPlay;
  const context = vm.createContext({
    speechSeq: 1, speaker: {}, console,
    URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} },
    Audio: class { play() { return new Promise((resolve, reject) => { rejectPlay = reject; }); } removeAttribute() {} load() {} remove() {} },
    document: { body: { appendChild() {} } },
    mediaErrorText: () => '',
  });
  vm.runInContext(section('    function playViaElement(', '    async function playServerAudio('), context);
  const playing = context.playViaElement({}, 'audio/wav');
  context.speechSeq++;
  rejectPlay({ name: 'AbortError', message: 'cancelled' });
  const result = await playing;
  assert.equal(result.ok, true);
  assert.equal(result.interrupted, true);
});

test('call starts microphone immediately in parallel with the voice channel', async () => {
  const actions = [];
  let resolveSocket;
  const context = vm.createContext({
    app: { micEnabled: false },
    voice: { reconnectTimer: null },
    ui: { phoneInput: { value: '+79990000001', focus() {} } },
    micStream: { readyMs: 0 }, performance, clearTimeout, console,
    closePopover: () => {}, abortUtterance: () => {}, closeCallMic: () => {}, stopSpeech: () => {},
    syncPhonePresetsActive: () => {}, clearTranscript: () => {}, startCallTimer: () => {},
    refreshLiveView: () => {}, record: () => {}, addCallStartPending: () => ({}),
    setBusy: () => {}, showStatus: () => {}, warmServer: () => {},
    prepareAudioContextSync: () => actions.push('prepare'),
    unlockAudio: () => actions.push('unlock'),
    openCallMic: ({ allowPrompt }) => {
      assert.equal(allowPrompt, true);
      assert.equal(context.app.micEnabled, true);
      actions.push('microphone');
      return Promise.resolve();
    },
    openWsChannel: () => {
      actions.push('websocket');
      return new Promise(resolve => { resolveSocket = resolve; });
    },
  });
  vm.runInContext(section('    async function startCall()', '    // Завершает звонок'), context);
  const starting = context.startCall();
  assert.deepEqual(actions, ['prepare', 'unlock', 'microphone', 'websocket']);
  resolveSocket();
  await starting;
});

test('a failed voice channel disables the automatically started microphone', async () => {
  let closedWithMicOn = null;
  const context = vm.createContext({
    app: { micEnabled: false }, voice: { reconnectTimer: null },
    ui: { phoneInput: { value: '123', focus() {} } }, micStream: { readyMs: 0 },
    performance, clearTimeout, console,
    closePopover: () => {}, abortUtterance: () => {},
    closeCallMic: () => { closedWithMicOn = context.app.micEnabled; },
    stopSpeech: () => {}, syncPhonePresetsActive: () => {}, clearTranscript: () => {},
    startCallTimer: () => {}, stopCallTimer: () => {}, refreshLiveView: () => {},
    record: () => {}, addCallStartPending: () => ({}), setBusy: () => {}, showStatus: () => {},
    warmServer: () => {}, prepareAudioContextSync: () => {}, unlockAudio: () => {},
    openCallMic: () => Promise.resolve(), openWsChannel: () => Promise.reject(new Error('offline')),
    closeVoiceChannel: () => {}, resetAudioContext: () => {},
  });
  vm.runInContext(section('    async function startCall()', '    // Завершает звонок'), context);
  await context.startCall();
  assert.equal(context.app.micEnabled, false);
  assert.equal(closedWithMicOn, false);
  assert.equal(context.app.sessionActive, false);
});

test('PCM resampling preserves phase across buffers at common input rates', () => {
  const context = vm.createContext({ Float32Array, Number, Math, Error });
  vm.runInContext(section('    function resample(', '    function percentileOf('), context);
  function convert(samples, from, chunkSize) {
    const state = { total: 0, nextPos: 0, history: null };
    const output = [];
    for (let i = 0; i < samples.length; i += chunkSize) {
      output.push(...context.resample(samples.subarray(i, i + chunkSize), from, 16000, state));
    }
    return output;
  }
  for (const from of [8000, 16000, 44100, 48000]) {
    const samples = Float32Array.from({ length: from }, (_, i) => Math.sin(2 * Math.PI * 1000 * i / from));
    const whole = convert(samples, from, samples.length);
    const chunks = convert(samples, from, 173);
    assert.equal(whole.length, 16000, `rate ${from}`);
    assert.equal(chunks.length, whole.length);
    for (let i = 0; i < whole.length; i++) {
      assert.ok(Number.isFinite(chunks[i]));
      assert.ok(Math.abs(whole[i] - chunks[i]) < 1e-6, `rate ${from}, sample ${i}`);
    }
  }
});

test('resampling attenuates frequencies above the output Nyquist frequency', () => {
  const context = vm.createContext({ Float32Array, Number, Math, Error });
  vm.runInContext(section('    function resample(', '    function percentileOf('), context);
  function amplitude(freq) {
    const samples = Float32Array.from({ length: 48000 }, (_, i) => Math.sin(2 * Math.PI * freq * i / 48000));
    const state = { total: 0, nextPos: 0, history: null };
    const output = context.resample(samples, 48000, 16000, state).subarray(100);
    return Math.sqrt(output.reduce((sum, x) => sum + x * x, 0) / output.length);
  }
  const pass = amplitude(1000), stop = amplitude(11000);
  assert.ok(pass > 0.6 && pass < 0.8, `passband RMS ${pass}`);
  assert.ok(stop / pass < 0.03, `stopband RMS ${stop}`);
});

test('failed or congested audio sends abort the utterance instead of completing a truncated phrase', () => {
  let now = 5000;
  const sent = [];
  const socket = { readyState: 1, bufferedAmount: 0, send(x) { sent.push(x); }, close() {} };
  const ui = { status: { textContent: '' } };
  const context = vm.createContext({
    WebSocket: { OPEN: 1 }, WS_MAX_AUDIO_BUFFER_BYTES: 256 * 1024,
    SAMPLE_RATE: 16000, MIN_AUDIO_SEC: 0.3, MAX_UTTERANCE_MS: 20000,
    voice: { ws: socket }, ui, console,
    performance: { now: () => now }, Date,
    micStream: { streaming: false, sentBytes: 0, voicedMs: 0, preroll: [], prerollBytes: 0, sendBlockedUntil: 0, sendError: '' },
    showStatus: text => { ui.status.textContent = text; }, setVoiceState() {},
    micListening: () => true, floatToInt16: frame => new Int16Array(frame), pushPreroll() {},
  });
  vm.runInContext(section('    function voiceSend(payload)', '    function setVoiceState('), context);
  vm.runInContext(section('    function failAudioSend(', '    /* =========================================================\n       Композер'), context);
  const controls = () => sent.filter(x => typeof x === 'string').map(x => JSON.parse(x).t);
  context.startUtterance();
  context.onAudioFrame(new Int16Array(5000));
  context.endUtterance();
  assert.deepEqual(controls(), ['speech_start', 'utterance_end']);
  sent.length = 0;
  now += 2000;
  context.micStream.preroll = [new Int16Array(20)];
  socket.bufferedAmount = 256 * 1024;
  context.startUtterance();
  assert.deepEqual(controls(), ['speech_start', 'abort']);
  assert.equal(context.micStream.streaming, false);
  assert.match(ui.status.textContent, /повторите фразу/);
  sent.length = 0;
  now += 2000;
  socket.bufferedAmount = 0;
  context.startUtterance();
  socket.bufferedAmount = 256 * 1024;
  context.onAudioFrame(new Int16Array(20));
  context.endUtterance();
  assert.deepEqual(controls(), ['speech_start', 'abort']);
});

test('heartbeat detects a confirmed pong server that stops responding and ignores background pauses', () => {
  let now = 0, sends = 0, aborts = 0, reconnects = 0, closes = 0;
  const document = { hidden: false };
  const socket = { abandoned: false, close: () => closes++ };
  const voice = { ws: socket, ready: true, pingTimer: null };
  const context = vm.createContext({
    voice, app: { ending: false }, document, console, clearInterval,
    performance: { now: () => now }, WS_HEARTBEAT_INTERVAL_MS: 15000, WS_HEARTBEAT_TIMEOUT_MS: 60000,
    inCall: () => true, voiceSend: () => { sends++; return true; },
    abortUtterance: () => aborts++, setChannelLabel() {}, showStatus() {}, reconnectVoice: () => reconnects++,
  });
  vm.runInContext(section('    function recoverStalledVoice(', '    function openWsChannel('), context);
  let beat = { supported: false, sentAt: null, lastTick: null };
  for (now = 15000; now <= 150000; now += 15000) context.heartbeatTick(socket, beat);
  assert.equal(reconnects, 0, 'no forced timeout until server has confirmed pong support');
  assert.equal(sends, 10);
  beat.supported = true;
  context.heartbeatTick(socket, beat);
  for (let i = 0; i < 4; i++) { now += 15000; context.heartbeatTick(socket, beat); }
  assert.equal(reconnects, 1);
  assert.equal(closes, 1);
  assert.equal(aborts, 1);
  const next = { abandoned: false, close() {} };
  voice.ws = next;
  voice.ready = true;
  beat = { supported: true, sentAt: null, lastTick: null };
  now = 0;
  context.heartbeatTick(next, beat);
  document.hidden = true;
  now = 15000;
  context.heartbeatTick(next, beat);
  document.hidden = false;
  now = 105000;
  context.heartbeatTick(next, beat);
  assert.equal(reconnects, 1, 'background gap must not cause timeout');
  assert.equal(beat.sentAt, now);
});

test('audio player is attached while playing and removed after completion', async () => {
  let last, revoked = 0;
  class FakeAudio {
    constructor() { last = this; this.isConnected = false; }
    play() { assert.equal(this.isConnected, true); return Promise.resolve(); }
    removeAttribute() {}
    load() {}
    remove() { this.isConnected = false; }
  }
  const context = vm.createContext({
    speechSeq: 1, speaker: {}, console, Audio: FakeAudio,
    URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => revoked++ },
    document: { body: { appendChild: el => { el.isConnected = true; } } },
    mediaErrorText: () => '',
  });
  vm.runInContext(section('    function playViaElement(', '    async function playServerAudio('), context);
  const result = context.playViaElement({}, 'audio/wav');
  assert.equal(last.isConnected, true);
  last.onended();
  assert.equal((await result).ok, true);
  assert.equal(last.isConnected, false);
  assert.equal(revoked, 1);
});

test('starting a new speech settles the previous speech as interrupted', async () => {
  let begins = 0, ends = 0;
  const speaker = { finish: null };
  const context = vm.createContext({
    speechSeq: 0, speaker, app: { ttsSpeaker: null }, window: {}, console,
    stopSpeechWatchdog() {}, releaseSpeaker() {}, beginSpeaking: () => begins++, endSpeaking: () => ends++,
    playServerAudio: () => new Promise(resolve => { speaker.finish = resolve; }),
    speakWithBrowser: () => { throw new Error('unexpected browser fallback'); },
  });
  vm.runInContext(section('    async function speak(reply, customLabel)', '    /* =========================================================\n       Переключение режимов'), context);
  const old = context.speak({ audio_b64: 'old' });
  const next = context.speak({ audio_b64: 'new' });
  assert.equal((await old).interrupted, true);
  assert.equal(begins, 2);
  assert.equal(ends, 0, 'old speech must not end the new speech');
  speaker.finish({ ok: true });
  assert.equal((await next).ok, true);
  assert.equal(ends, 1);
});

test('late ASR result does not overwrite a newer result', async () => {
  const requests = [], history = [];
  let result = '';
  const asr = { recording: true, pending: false, recognition: null, chunks: [new Int16Array(6000)], resultRevision: 1 };
  const context = vm.createContext({
    asr, SAMPLE_RATE: 16000, MIN_AUDIO_SEC: 0.3, app: { cfg: { asr: 'test' } }, performance, Blob, console,
    stopAsrCapture: () => { asr.recording = false; },
    setAsrResult: text => { result = text; }, addAsrHistory: text => history.push(text),
    api: () => new Promise(resolve => requests.push(resolve)),
  });
  vm.runInContext(section('    async function stopAsrAndSend()', '    function startBrowserAsr()'), context);
  const first = context.stopAsrAndSend();
  asr.resultRevision++;
  asr.recording = true;
  asr.chunks = [new Int16Array(6000)];
  const second = context.stopAsrAndSend();
  requests[1]({ text: 'новый' });
  await second;
  requests[0]({ text: 'старый' });
  await first;
  assert.equal(result, 'новый');
  assert.deepEqual(history, ['новый']);
});

test('TTS stopped while a request is pending does not play its late response', async () => {
  let resolveRequest, plays = 0, status = '';
  const context = vm.createContext({
    speechSeq: 0, ui: { ttsInput: { value: 'тест' }, ttsPlayLabel: { textContent: '' } },
    app: { ttsSpeaker: null },
    unlockAudio() {}, showTtsStatus: text => { status = text; }, stopSpeech() {},
    postJson: body => body, addTtsHistory() {},
    api: () => new Promise(resolve => { resolveRequest = resolve; }),
    speak: () => { plays++; return Promise.resolve({ ok: true }); },
  });
  vm.runInContext(section('    let ttsPlaying = false;', '    function addTtsHistory('), context);
  const pending = context.playTts();
  await context.playTts();
  resolveRequest({ audio_b64: 'late', audio_mime: 'audio/wav' });
  await pending;
  assert.equal(plays, 0);
  assert.equal(status, 'Остановлено');
});

test('starting a new ASR probe invalidates a previous result before requesting microphone access', async () => {
  const asr = { recording: false, pending: false, recognition: null, resultRevision: 4 };
  const context = vm.createContext({
    asr, app: { cfg: { asr: 'cloudru' } }, micSupported: false,
    window: { isSecureContext: true }, setAsrResult() {},
  });
  vm.runInContext(section('    async function startAsr()', '    function stopAsrCapture()'), context);
  await context.startAsr();
  assert.equal(asr.resultRevision, 5);
});

test('text interruption fades playback while sending barge and text immediately', () => {
  const calls = [];
  const context = vm.createContext({
    app: { speaking: true }, voice: { generation: 3, turn: 5 }, speechSeq: 2,
    speaker: { el: null, source: null, quietWavSeq: -1 }, ArrayBuffer, DataView, setTimeout,
    inCall: () => true, composerLocked: () => false,
    speechPlayedMs: () => 1234,
    stopSpeechWatchdog: () => calls.push('watchdog'),
    fadeOutSpeech: ms => calls.push(['fade', ms]),
    endSpeaking: () => { calls.push('ended'); context.app.speaking = false; },
    voiceSend: event => { calls.push(event.t); return true; },
    abortUtterance: () => calls.push('abort'),
    addBubble: () => calls.push('bubble'),
    setVoiceState: state => calls.push(state),
  });
  vm.runInContext(section('    function quietCutDelay(', '    function fadeOutSpeech('), context);
  vm.runInContext(section('    function sendText(text)', '    /* =========================================================\n       Непрерывный захват'), context);
  context.sendText(' привет ');
  assert.deepEqual(calls, ['watchdog', ['fade', 240], 'ended', 'barge', 'abort', 'bubble', 'text', 'thinking']);
  assert.equal(context.voice.generation, 4);
  assert.equal(context.speechSeq, 3);
  calls.length = 0;
  context.sendText('следующая');
  assert.deepEqual(calls, ['abort', 'bubble', 'text', 'thinking']);
});

test('quiet cut uses valid PCM data and rejects malformed or unsupported WAV', () => {
  const context = vm.createContext({ ArrayBuffer, DataView });
  vm.runInContext(section('    function quietCutDelay(', '    function fadeInterruptionAtPause('), context);
  function wav() {
    const rate = 16000, frames = rate, bytes = new ArrayBuffer(44 + frames * 2), v = new DataView(bytes);
    function tag(offset, value) { for (let i = 0; i < 4; i++) v.setUint8(offset + i, value.charCodeAt(i)); }
    tag(0, 'RIFF'); v.setUint32(4, bytes.byteLength - 8, true); tag(8, 'WAVE');
    tag(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    tag(36, 'data'); v.setUint32(40, frames * 2, true);
    for (let i = 0; i < frames; i++) v.setInt16(44 + i * 2, (i >= 3200 && i < 4160) ? 0 : 10000, true);
    return bytes;
  }
  const bytes = wav();
  assert.equal(context.quietCutDelay(bytes, 0), 200);
  assert.equal(context.quietCutDelay(bytes, 400), 720);
  // Тишина 30 мс внутри слова недостаточна; берём следующую паузу 70 мс.
  const altered = new DataView(bytes);
  for (let i = 0; i < 16000; i++) {
    const silent = (i >= 3200 && i < 3680) || (i >= 8000 && i < 9120);
    altered.setInt16(44 + i * 2, silent ? 0 : 10000, true);
  }
  assert.equal(context.quietCutDelay(bytes, 0), 500);
  assert.equal(context.quietCutDelay(new ArrayBuffer(10), 0), null);
  new DataView(bytes).setUint16(22, 2, true);
  assert.equal(context.quietCutDelay(bytes, 0), null);
});

test('delayed quiet cut never stops a newer player', () => {
  const timers = [], fades = [];
  const old = { currentTime: 1 };
  const context = vm.createContext({
    speaker: { el: old, source: null, quietWav: {}, quietWavSeq: 5 }, speechSeq: 5,
    quietCutDelay: () => 120, speechPlayedMs: () => 1000,
    stopSpeechWatchdog: () => {}, fadeOutSpeech: ms => fades.push(ms),
    setTimeout: callback => timers.push(callback),
  });
  vm.runInContext(section('    function fadeInterruptionAtPause(', '    function fadeOutSpeech('), context);
  context.fadeInterruptionAtPause(240);
  assert.equal(context.speechSeq, 6);
  assert.equal(timers.length, 1);
  context.speechSeq++;
  context.speaker.el = { currentTime: 0 };
  timers[0]();
  assert.deepEqual(fades, []);
  context.speaker.el = old;
  context.speaker.quietWavSeq = context.speechSeq;
  context.fadeInterruptionAtPause(240);
  timers[1]();
  assert.deepEqual(fades, [80]);
});

test('voice interruption starts listening immediately and leaves delayed audio stop isolated', () => {
  const calls = [], timers = [];
  const old = { currentTime: 1.2 };
  const context = vm.createContext({
    app: { speaking: true }, voice: { generation: 2, turn: 5 }, speechSeq: 7,
    speaker: { el: old, source: null, quietWav: {}, quietWavSeq: 7 },
    micStream: { streaming: false, lastVoiceAt: 0 }, Date,
    quietCutDelay: () => 500, speechPlayedMs: () => 1200,
    stopSpeechWatchdog: () => calls.push('watchdog'),
    fadeOutSpeech: ms => calls.push(['fade', ms]),
    endSpeaking: () => { calls.push('ended'); context.app.speaking = false; },
    voiceSend: event => { calls.push(event.t); return true; },
    record: () => calls.push('record'), setVoiceState: state => calls.push(state),
    startUtterance: () => calls.push('start-audio'),
    setTimeout: (callback, delay) => { timers.push({ callback, delay }); },
  });
  vm.runInContext(section('    function fadeInterruptionAtPause(', '    function fadeOutSpeech('), context);
  vm.runInContext(section('    function softBarge()', '    function voiceUrl()'), context);
  context.softBarge();
  assert.deepEqual(calls, ['watchdog', 'ended', 'barge', 'record', 'listening', 'start-audio']);
  assert.equal(timers[0].delay, 500);
  assert.equal(context.voice.generation, 3);
  timers[0].callback();
  assert.deepEqual(calls.at(-1), ['fade', 80]);
});

test('voice interruption retains 150ms fallback without Cloud.ru WAV', () => {
  const fades = [];
  const context = vm.createContext({
    speaker: { el: null, source: null, quietWavSeq: -1 }, speechSeq: 4,
    speechPlayedMs: () => 100, stopSpeechWatchdog() {},
    fadeOutSpeech: ms => fades.push(ms), setTimeout,
  });
  vm.runInContext(section('    function fadeInterruptionAtPause(', '    function fadeOutSpeech('), context);
  context.fadeInterruptionAtPause(150);
  assert.deepEqual(fades, [150]);
});
