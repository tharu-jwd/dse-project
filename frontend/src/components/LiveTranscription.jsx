import { useEffect, useRef, useState } from 'react'
import { api, API_BASE_URL } from '../api'
import { useLanguage } from '../contexts/LanguageContext'
import Icon from './Icon'
import { Alert } from './UI'

const TARGET_SAMPLE_RATE = 16000
const CHUNK_SAMPLES = TARGET_SAMPLE_RATE * 0.25 // ~250ms per chunk, per the streaming protocol

function downsampleTo16k(input, inputSampleRate) {
  if (inputSampleRate === TARGET_SAMPLE_RATE) return input
  const ratio = inputSampleRate / TARGET_SAMPLE_RATE
  const outLength = Math.floor(input.length / ratio)
  const output = new Float32Array(outLength)
  for (let i = 0; i < outLength; i++) {
    output[i] = input[Math.floor(i * ratio)]
  }
  return output
}

function floatTo16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length)
  for (let i = 0; i < float32Array.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32Array[i]))
    output[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
  }
  return output
}

function wsUrl(path) {
  return `${API_BASE_URL.replace(/^http/, 'ws')}${path}`
}

// dangerouslySetInnerHTML needs its input pre-escaped - a transcribed word
// is never expected to contain markup, but user-editable text shouldn't
// ever be trusted to skip this regardless of the source.
function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

/**
 * Live note-taking: streams mic audio to the streaming WebSocket and
 * renders partial (grey) and final (black) transcript text as it arrives.
 */
const VOICE_BAR_COUNT = 9
const VOICE_THRESHOLD = 0.02

export default function LiveTranscription({
  title,
  onSessionEnd,
  disabled = false,
  mode = 'NOTE',
  autoStart = false,
}) {
  const { t } = useLanguage()
  const [status, setStatus] = useState('idle') // idle | connecting | recording | stopping | error
  const [error, setError] = useState('')
  const [finals, setFinals] = useState([]) // [{ segment, segmentId, text, dirty }]
  const [partial, setPartial] = useState('')
  const [seconds, setSeconds] = useState(0)
  const [voiceDetected, setVoiceDetected] = useState(false)
  const [commandFeedback, setCommandFeedback] = useState('')
  const [commandFeedbackTone, setCommandFeedbackTone] = useState('success')
  const [saveState, setSaveState] = useState('idle') // idle | saving | saved | error

  const transcriptIdRef = useRef(null)
  const wsRef = useRef(null)
  const audioCtxRef = useRef(null)
  const streamRef = useRef(null)
  const processorRef = useRef(null)
  const pendingSamplesRef = useRef(new Float32Array(0))
  const intervalRef = useRef(null)
  const endedRef = useRef(false)
  const statusRef = useRef('idle')
  const levelRef = useRef(0)
  const voiceDetectedRef = useRef(false)
  const barRefs = useRef([])
  const meterFrameRef = useRef(null)
  const feedbackTimeoutRef = useRef(null)

  const showCommandFeedback = (message, tone = 'success') => {
    window.clearTimeout(feedbackTimeoutRef.current)
    setCommandFeedback(message)
    setCommandFeedbackTone(tone)
    feedbackTimeoutRef.current = window.setTimeout(() => setCommandFeedback(''), 2200)
  }

  useEffect(() => {
    statusRef.current = status
  }, [status])

  const stopMeterLoop = () => {
    if (meterFrameRef.current) window.cancelAnimationFrame(meterFrameRef.current)
    meterFrameRef.current = null
    levelRef.current = 0
    voiceDetectedRef.current = false
    setVoiceDetected(false)
    barRefs.current.forEach((bar) => {
      if (bar) bar.style.transform = 'scaleY(0.15)'
    })
  }

  const startMeterLoop = () => {
    const tick = () => {
      const level = levelRef.current
      barRefs.current.forEach((bar, index) => {
        if (!bar) return
        // Give each bar a slightly different sensitivity so the meter looks
        // organic rather than every bar moving in lockstep.
        const wobble = 0.6 + ((index % 4) / 4) * 0.8
        const scale = Math.min(1, 0.15 + level * wobble * 4)
        bar.style.transform = `scaleY(${scale})`
      })
      const detected = level > VOICE_THRESHOLD
      if (detected !== voiceDetectedRef.current) {
        voiceDetectedRef.current = detected
        setVoiceDetected(detected)
      }
      meterFrameRef.current = window.requestAnimationFrame(tick)
    }
    meterFrameRef.current = window.requestAnimationFrame(tick)
  }

  const cleanupAudio = () => {
    processorRef.current?.disconnect()
    processorRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => {})
    }
    audioCtxRef.current = null
    window.clearInterval(intervalRef.current)
    stopMeterLoop()
  }

  useEffect(
    () => () => {
      cleanupAudio()
      window.clearTimeout(feedbackTimeoutRef.current)
    },
    [],
  )

  const start = async () => {
    setError('')
    setFinals([])
    setPartial('')
    setSeconds(0)
    endedRef.current = false

    if (!title.trim()) {
      setError(t('live.titleRequired'))
      return
    }
    if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
      setError(t('live.notSupported'))
      return
    }

    setStatus('connecting')

    const token = localStorage.getItem('sinhaspeech_token')
    const socket = new WebSocket(wsUrl(`/streaming/ws?token=${encodeURIComponent(token || '')}`))
    socket.binaryType = 'arraybuffer'
    wsRef.current = socket

    socket.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        streamRef.current = stream

        const AudioCtx = window.AudioContext || window.webkitAudioContext
        const audioCtx = new AudioCtx()
        audioCtxRef.current = audioCtx

        const source = audioCtx.createMediaStreamSource(stream)
        const processor = audioCtx.createScriptProcessor(4096, 1, 1)
        processorRef.current = processor
        const silentGain = audioCtx.createGain()
        silentGain.gain.value = 0

        processor.onaudioprocess = (event) => {
          if (socket.readyState !== WebSocket.OPEN) return
          const input = event.inputBuffer.getChannelData(0)

          let sumSquares = 0
          for (let i = 0; i < input.length; i++) sumSquares += input[i] * input[i]
          const rms = Math.sqrt(sumSquares / input.length)
          // Exponential moving average so the meter eases rather than jitters.
          levelRef.current = levelRef.current * 0.6 + rms * 0.4

          const downsampled = downsampleTo16k(input, audioCtx.sampleRate)

          const merged = new Float32Array(pendingSamplesRef.current.length + downsampled.length)
          merged.set(pendingSamplesRef.current)
          merged.set(downsampled, pendingSamplesRef.current.length)

          let offset = 0
          while (merged.length - offset >= CHUNK_SAMPLES) {
            const slice = merged.subarray(offset, offset + CHUNK_SAMPLES)
            socket.send(floatTo16BitPCM(slice).buffer)
            offset += CHUNK_SAMPLES
          }
          pendingSamplesRef.current = merged.slice(offset)
        }

        source.connect(processor)
        processor.connect(silentGain)
        silentGain.connect(audioCtx.destination)

        socket.send(JSON.stringify({ type: 'start', title: title.trim(), mode }))
        setStatus('recording')
        intervalRef.current = window.setInterval(() => setSeconds((value) => value + 1), 1000)
        startMeterLoop()
      } catch (cause) {
        setStatus('error')
        if (cause?.name === 'NotAllowedError' || cause?.name === 'SecurityError')
          setError(t('live.permissionDenied'))
        else if (cause?.name === 'NotFoundError') setError(t('live.noMicrophone'))
        else setError(t('live.startFailed'))
        socket.close()
      }
    }

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data)
      if (message.type === 'partial') {
        setPartial(message.text)
      } else if (message.type === 'final') {
        transcriptIdRef.current = message.transcript_id
        setFinals((prev) => [
          ...prev,
          { segment: message.segment, segmentId: message.segment_id, text: message.text, dirty: false },
        ])
        setPartial('')
      } else if (message.type === 'command') {
        setPartial('')
        if (message.command === 'delete') {
          setFinals((prev) => {
            if (prev.length === 0) {
              showCommandFeedback(t('live.nothingToDelete'))
              return prev
            }
            showCommandFeedback(t('live.lastLineDeleted'))
            return prev.slice(0, -1)
          })
        } else if (message.command === 'stop') {
          showCommandFeedback(t('live.stopping'))
          stop()
        }
      } else if (message.type === 'command_maybe') {
        // Ambiguous: neither signal was confident enough to act on its
        // own, or the two disagreed. Nothing was changed - the words are
        // kept as ordinary text - this is just a heads-up in case the
        // student meant a command and wants to say it again more clearly.
        const candidate = message.fuzzy_command || message.embedding_command
        showCommandFeedback(
          candidate ? t('live.didYouMean', candidate) : t('live.possibleCommand'),
          'advisory',
        )
      } else if (message.type === 'session_end') {
        endedRef.current = true
        cleanupAudio()
        setStatus('idle')
        // Don't hand off to the review screen with edits still sitting
        // unsaved in local state - flush them first (saveEdits reads
        // `finals` fresh via its functional setState, so this is safe to
        // fire from inside the same event that just finished the last
        // setFinals() call for this session).
        saveEdits().finally(() => onSessionEnd(message.transcript_id))
      } else if (message.type === 'error') {
        setError(message.message)
      }
    }

    socket.onclose = () => {
      cleanupAudio()
      if (!endedRef.current) {
        const wasStopping = statusRef.current === 'stopping'
        setStatus(wasStopping ? 'idle' : 'error')
        if (!wasStopping) setError((prev) => prev || t('live.connectionLost'))
      }
    }

    socket.onerror = () => {
      setStatus('error')
      setError(t('live.connectFailed'))
    }
  }

  // Voice-driven handoff: the quiz page's always-on command mic sets
  // `autoStart` once it hears "answer", so this session starts itself
  // instead of waiting for a click - it only fires once per handoff
  // (guarded on `status === 'idle'`), not on every re-render.
  useEffect(() => {
    if (autoStart && status === 'idle' && !disabled) start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart])

  const stop = () => {
    setStatus('stopping')
    const socket = wsRef.current
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'stop' }))
    } else {
      cleanupAudio()
      setStatus('idle')
    }
  }

  const saveEdits = async () => {
    const transcriptId = transcriptIdRef.current
    const changed = finals.filter((item) => item.dirty && item.segmentId)
    if (!transcriptId || !changed.length) return

    setSaveState('saving')
    try {
      await api.updateTranscript(transcriptId, {
        segments: changed.map((item) => ({ id: item.segmentId, text: item.text })),
      })
      setFinals((prev) =>
        prev.map((item) => (item.dirty ? { ...item, dirty: false } : item)),
      )
      setSaveState('saved')
    } catch (cause) {
      setSaveState('error')
      setError(cause.message || t('live.editsNotSaved'))
    }
  }

  // Fires on leaving a segment span the student just clicked into and
  // edited. Deliberately doesn't go through saveEdits (which reads
  // `finals` from this render's closure) - a contentEditable span
  // only commits to React state here, on blur, never on every keystroke
  // (that would fight the browser's own cursor position on each render),
  // so at blur time React state may not have caught up yet. The DOM's own
  // textContent is the actual source of truth for what the student typed.
  const handleSegmentBlur = async (item, event) => {
    const text = event.currentTarget.textContent
    if (text === item.text) return

    const transcriptId = transcriptIdRef.current
    setFinals((prev) =>
      prev.map((entry) => (entry.segment === item.segment ? { ...entry, text, dirty: true } : entry)),
    )

    if (!transcriptId || !item.segmentId) return

    setSaveState('saving')
    try {
      await api.updateTranscript(transcriptId, {
        segments: [{ id: item.segmentId, text }],
      })
      setFinals((prev) =>
        prev.map((entry) => (entry.segment === item.segment ? { ...entry, dirty: false } : entry)),
      )
      setSaveState('saved')
    } catch (cause) {
      setSaveState('error')
      setError(cause.message || t('live.editsNotSaved'))
    }
  }

  const time = (value) =>
    `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`

  const isRecording = status === 'recording' || status === 'stopping'
  const hasUnsavedEdits = finals.some((item) => item.dirty)

  return (
    <div className="note-workspace">
      {error && <Alert>{error}</Alert>}
      <div className={`note-toolbar ${isRecording ? 'is-recording' : ''}`}>
        <button
          type="button"
          className={`note-toolbar__record ${isRecording ? 'is-recording' : ''}`}
          onClick={isRecording ? stop : start}
          disabled={disabled || status === 'connecting' || status === 'stopping'}
          aria-label={isRecording ? t('live.stopRecording') : t('live.startLive')}
        >
          <Icon name={isRecording ? 'stop' : 'mic'} size={19} />
        </button>
        <div className={`voice-meter ${voiceDetected ? 'voice-meter--active' : ''}`}>
          {Array.from({ length: VOICE_BAR_COUNT }).map((_, index) => (
            <i key={index} ref={(el) => (barRefs.current[index] = el)} />
          ))}
        </div>
        <span className="note-toolbar__status" aria-live="polite">
          {status === 'connecting' && t('live.connecting')}
          {status === 'stopping' && t('live.finishing')}
          {status === 'recording' && (voiceDetected ? t('live.voiceDetected') : t('live.listening'))}
          {(status === 'idle' || status === 'error') && t('live.readyToRecord')}
        </span>
        {isRecording && <time className="note-toolbar__time">{time(seconds)}</time>}
      </div>
      {commandFeedback && (
        <p className={`command-feedback command-feedback--${commandFeedbackTone}`} role="status">
          <Icon name={commandFeedbackTone === 'advisory' ? 'alert' : 'check'} size={14} />{' '}
          {commandFeedback}
        </p>
      )}
      {finals.length > 0 && (
        <div className="note-save-bar">
          <span className="note-save-bar__status" role="status" aria-live="polite">
            {saveState === 'saving'
              ? t('live.savingEdits')
              : hasUnsavedEdits
                ? t('live.unsavedEdits')
                : saveState === 'saved'
                  ? t('live.allEditsSaved')
                  : t('live.clickToEdit')}
          </span>
          <button
            type="button"
            className="button button--secondary button--small"
            onClick={saveEdits}
            disabled={!hasUnsavedEdits || saveState === 'saving'}
          >
            {saveState === 'saving' ? (
              <span className="spinner spinner--small" />
            ) : (
              <Icon name="check" size={15} />
            )}{' '}
            {t('live.saveEdits')}
          </button>
        </div>
      )}
      <div className="note-page note-page--editable" aria-live="polite">
        {finals.length === 0 && !partial ? (
          <p className="note-page__placeholder">
            {isRecording ? t('live.listeningForSpeech') : t('live.notePlaceholder')}
          </p>
        ) : (
          <p className="note-page__paragraph" lang="si">
            {finals.map((item) => (
              <span key={item.segment}>
                <span
                  className={`note-page__segment ${item.dirty ? 'is-dirty' : ''}`}
                  contentEditable
                  suppressContentEditableWarning
                  role="textbox"
                  aria-multiline="false"
                  aria-label={t('live.noteLine', item.segment + 1)}
                  onBlur={(e) => handleSegmentBlur(item, e)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') e.preventDefault()
                  }}
                  dangerouslySetInnerHTML={{ __html: escapeHtml(item.text) }}
                />{' '}
              </span>
            ))}
            {partial && <span className="live-transcript__partial">{partial}</span>}
          </p>
        )}
      </div>
    </div>
  )
}
