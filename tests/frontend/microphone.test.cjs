const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('src/assistant/api/static/index.html', 'utf8');
function section(start, end) {
  const a = html.indexOf(start), b = html.indexOf(end, a);
  assert.ok(a >= 0 && b > a);
  return html.slice(a, b);
}

test('stopping a microphone track does not close or replace the healthy audio context', () => {
  let created = 0, closed = 0, stopped = 0;
  class FakeContext {
    constructor() { created++; this.state = 'running'; }
    createMediaStreamSource() {}
    createScriptProcessor() {}
    createGain() {}
    close() { closed++; return Promise.resolve(); }
  }
  const context = vm.createContext({
    window: { AudioContext: FakeContext },
    mic: {},
    micStream: { stream: { getTracks: () => [{ stop: () => stopped++ }] } },
  });
  vm.runInContext(section('    let audioCtx = null;', '    async function tryResumeAudioContext'), context);
  vm.runInContext(section('    function releaseMic()', '    function disconnectCaptureNodes()'), context);
  const first = context.prepareAudioContextSync(true);
  context.releaseMic();
  const second = context.prepareAudioContextSync(true);
  assert.equal(first, second);
  assert.equal(created, 1);
  assert.equal(closed, 0);
  assert.equal(stopped, 1);
});

test('default microphone is opened once without stopping a probe stream', async () => {
  let opens = 0;
  const stream = {};
  const context = vm.createContext({
    mic: { deviceId: '', granted: false },
    micAudioConstraints: () => ({ echoCancellation: true }),
    navigator: { mediaDevices: { getUserMedia: async () => { opens++; return stream; } } },
    refreshMicDevices: async () => {}, localStorage: { removeItem() {} },
    performance, audioDebug: () => {}, console,
  });
  vm.runInContext(section('    async function timedGetUserMedia(', '    /* =========================================================\n       Транскрипт'), context);
  assert.equal(await context.micGetUserMedia(), stream);
  assert.equal(opens, 1);
});

test('invalid selected microphone falls back without reopening the successful stream', async () => {
  let opens = 0, refreshes = 0;
  const stream = {};
  const context = vm.createContext({
    mic: { deviceId: 'missing', granted: true },
    micAudioConstraints: () => ({ echoCancellation: true }),
    navigator: { mediaDevices: { getUserMedia: async () => {
      opens++;
      if (opens === 1) throw { name: 'NotFoundError' };
      return stream;
    } } },
    refreshMicDevices: async () => { refreshes++; }, localStorage: { removeItem() {} },
    performance, audioDebug: () => {}, console,
  });
  vm.runInContext(section('    async function timedGetUserMedia(', '    /* =========================================================\n       Транскрипт'), context);
  assert.equal(await context.micGetUserMedia(), stream);
  assert.equal(opens, 2);
  assert.equal(refreshes, 1);
});

test('stand keeps capture idle, explicitly releases it and cancels late opens', async () => {
  let stopped = 0, aborts = 0, resolveOpen;
  const button = { textContent: '', title: '', classList: { toggle() {} }, setAttribute() {}, addEventListener() {} };
  const ctx = vm.createContext({
    standMic: { keep: false, pending: false, token: 0 },
    micSupported: true, micStream: { stream: null }, ui: { standMicBtn: button },
    micTracksLive: s => Boolean(s), captureWanted: () => false,
    teardownCapture: () => { stopped++; ctx.micStream.stream = null; },
    prepareAudioContextSync() {},
    ensureCaptureOpen: () => new Promise(resolve => { resolveOpen = resolve; }),
    renderMicHint() {}, app: { micEnabled: true },
    abortUtterance: () => aborts++, closeCallMic() {}, stopAsrCapture() {},
    window: { addEventListener() {} },
  });
  vm.runInContext(section('    function stopCaptureIfIdle()', '    async function ensureCaptureOpen()'), ctx);
  const pending = ctx.holdStandMic();
  ctx.stopCaptureIfIdle();
  assert.equal(stopped, 0);
  ctx.releaseStandMic();
  assert.equal(stopped, 1);
  assert.equal(aborts, 1);
  assert.equal(ctx.app.micEnabled, false);
  resolveOpen();
  await pending;
  assert.equal(ctx.standMic.keep, false);
  assert.equal(ctx.standMic.pending, false);
  ctx.stopCaptureIfIdle();
  assert.equal(stopped, 2);
});

test('idle capture graph does not process or send microphone samples', () => {
  // Захват общий для worklet и script processor; проверяем обработчик после
  // выключения стартовой диагностики. Сам буфер может быть получен worklet.
  const callback = section('        const handleAudioFrame = (raw) => {', "        if (created.kind === 'worklet') {");
  let calculated = 0, sent = 0;
  const processor = {}, capture = { processor, framesSeen: 1 };
  const context = vm.createContext({
    processor, capture, epoch: 1, captureEpoch: 1, isCallMicLive: () => false,
    app: { mode: 'call' }, asr: { recording: false },
    performance: { now: () => 100 }, diagUntil: 0,
    rmsOf: () => { calculated++; return 0; }, onAudioFrame: () => sent++,
    previousConsumer: '', resampler: {}, audioDebug: () => {},
  });
  vm.runInContext(callback + '\nprocessor.handler = handleAudioFrame;', context);
  processor.handler(new Float32Array(32));
  assert.equal(calculated, 0);
  assert.equal(sent, 0);
});
