import { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from '../api'

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

export const VOICE_METER_BAR_COUNT = 9

/**
 * Opens a "COMMAND" mode streaming session: mic audio in, no dictation
 * text back - just recognized voice commands. Used on pages (transcript
 * review, quiz answers) where the student wants voice control but isn't
 * dictating a note. Mirrors the mic-capture pipeline in LiveTranscription,
 * which owns the NOTE-mode equivalent.
 */
export default function useVoiceCommands({ onCommand, onCommandMaybe } = {}) {
  const [status, setStatus] = useState('idle') // idle | connecting | listening | stopping | error
  const [error, setError] = useState('')
  const [voiceDetected, setVoiceDetected] = useState(false)

  const wsRef = useRef(null)
  const audioCtxRef = useRef(null)
  const streamRef = useRef(null)
  const processorRef = useRef(null)
  const pendingSamplesRef = useRef(new Float32Array(0))
  const statusRef = useRef('idle')
  const levelRef = useRef(0)
  const voiceDetectedRef = useRef(false)
  const meterFrameRef = useRef(null)
  const barRefs = useRef([])
  const registerBar = (index) => (el) => {
    barRefs.current[index] = el
  }
  const onCommandRef = useRef(onCommand)
  const onCommandMaybeRef = useRef(onCommandMaybe)

  useEffect(() => {
    onCommandRef.current = onCommand
  }, [onCommand])
  useEffect(() => {
    onCommandMaybeRef.current = onCommandMaybe
  }, [onCommandMaybe])
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
      const detected = level > 0.02
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
    stopMeterLoop()
  }

  useEffect(() => () => cleanupAudio(), [])

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

  const start = async () => {
    setError('')
    if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
      setError('Voice commands are not supported by this browser.')
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

        socket.send(JSON.stringify({ type: 'start', mode: 'COMMAND' }))
        setStatus('listening')
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
      if (message.type === 'command') {
        onCommandRef.current?.(message.command)
      } else if (message.type === 'command_maybe') {
        onCommandMaybeRef.current?.(message)
      } else if (message.type === 'session_end') {
        cleanupAudio()
        setStatus('idle')
      } else if (message.type === 'error') {
        setError(message.message)
      }
    }

    socket.onclose = () => {
      cleanupAudio()
      const wasStopping = statusRef.current === 'stopping'
      setStatus(wasStopping ? 'idle' : 'error')
      if (!wasStopping) setError((prev) => prev || 'The connection was lost.')
    }

    socket.onerror = () => {
      setStatus('error')
      setError('Could not connect to the streaming service.')
    }
  }

  const isListening = status === 'listening' || status === 'stopping'

  return { status, error, voiceDetected, isListening, start, stop, registerBar }
}
