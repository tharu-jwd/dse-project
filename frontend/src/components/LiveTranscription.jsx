import { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from '../api'
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

/**
 * Live note-taking: streams mic audio to the streaming WebSocket and
 * renders partial (grey) and final (black) transcript text as it arrives.
 */
const VOICE_BAR_COUNT = 9
const VOICE_THRESHOLD = 0.02

export default function LiveTranscription({ title, onSessionEnd, disabled = false }) {
  const [status, setStatus] = useState('idle') // idle | connecting | recording | stopping | error
  const [error, setError] = useState('')
  const [finals, setFinals] = useState([]) // [{ segment, text }]
  const [partial, setPartial] = useState('')
  const [seconds, setSeconds] = useState(0)
  const [voiceDetected, setVoiceDetected] = useState(false)

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

  useEffect(() => () => cleanupAudio(), [])

  const start = async () => {
    setError('')
    setFinals([])
    setPartial('')
    setSeconds(0)
    endedRef.current = false

    if (!title.trim()) {
      setError('Give your note a title before recording.')
      return
    }
    if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
      setError('Live transcription is not supported by this browser.')
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

        socket.send(JSON.stringify({ type: 'start', title: title.trim(), mode: 'NOTE' }))
        setStatus('recording')
        intervalRef.current = window.setInterval(() => setSeconds((value) => value + 1), 1000)
        startMeterLoop()
      } catch (cause) {
        setStatus('error')
        if (cause?.name === 'NotAllowedError' || cause?.name === 'SecurityError')
          setError('Microphone permission was denied.')
        else if (cause?.name === 'NotFoundError') setError('No microphone was found.')
        else setError('The microphone could not be started.')
        socket.close()
      }
    }

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data)
      if (message.type === 'partial') {
        setPartial(message.text)
      } else if (message.type === 'final') {
        setFinals((prev) => [...prev, { segment: message.segment, text: message.text }])
        setPartial('')
      } else if (message.type === 'session_end') {
        endedRef.current = true
        cleanupAudio()
        setStatus('idle')
        onSessionEnd(message.transcript_id)
      } else if (message.type === 'error') {
        setError(message.message)
      }
    }

    socket.onclose = () => {
      cleanupAudio()
      if (!endedRef.current) {
        const wasStopping = statusRef.current === 'stopping'
        setStatus(wasStopping ? 'idle' : 'error')
        if (!wasStopping) setError((prev) => prev || 'The connection was lost.')
      }
    }

    socket.onerror = () => {
      setStatus('error')
      setError('Could not connect to the streaming service.')
    }
  }

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

  const time = (value) =>
    `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`

  const isRecording = status === 'recording' || status === 'stopping'

  return (
    <div className="note-workspace">
      {error && <Alert>{error}</Alert>}
      <div className={`note-toolbar ${isRecording ? 'is-recording' : ''}`}>
        <button
          type="button"
          className={`note-toolbar__record ${isRecording ? 'is-recording' : ''}`}
          onClick={isRecording ? stop : start}
          disabled={disabled || status === 'connecting' || status === 'stopping'}
          aria-label={isRecording ? 'Stop recording' : 'Start live transcription'}
        >
          <Icon name={isRecording ? 'stop' : 'mic'} size={19} />
        </button>
        <div className={`voice-meter ${voiceDetected ? 'voice-meter--active' : ''}`}>
          {Array.from({ length: VOICE_BAR_COUNT }).map((_, index) => (
            <i key={index} ref={(el) => (barRefs.current[index] = el)} />
          ))}
        </div>
        <span className="note-toolbar__status" aria-live="polite">
          {status === 'connecting' && 'Connecting…'}
          {status === 'stopping' && 'Finishing…'}
          {status === 'recording' && (voiceDetected ? 'Voice detected' : 'Listening…')}
          {(status === 'idle' || status === 'error') && 'Ready to record'}
        </span>
        {isRecording && <time className="note-toolbar__time">{time(seconds)}</time>}
      </div>
      <div className="note-page" aria-live="polite">
        {finals.length === 0 && !partial ? (
          <p className="note-page__placeholder">
            {isRecording
              ? 'Listening for speech…'
              : 'Your note will appear here as you speak. Press the microphone above to begin.'}
          </p>
        ) : (
          <p>
            {finals.map((item) => (
              <span key={item.segment} className="live-transcript__final">
                {item.text}{' '}
              </span>
            ))}
            {partial && <span className="live-transcript__partial">{partial}</span>}
          </p>
        )}
      </div>
    </div>
  )
}
