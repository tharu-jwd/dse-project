import { useEffect, useId, useRef, useState } from 'react'
import FileUpload from './FileUpload'
import Icon from './Icon'
import { Alert } from './UI'

const time = (seconds) =>
  `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`

export default function AudioRecorder({ onUse, disabled = false }) {
  const uploadId = useId()
  const [state, setState] = useState('idle')
  const [seconds, setSeconds] = useState(0)
  const [audio, setAudio] = useState(null)
  const [error, setError] = useState('')
  const recorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])
  const intervalRef = useRef(null)
  const urlRef = useRef('')

  const releaseStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }
  const clearAudio = () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    urlRef.current = ''
    setAudio(null)
    setSeconds(0)
    setState('idle')
  }

  useEffect(
    () => () => {
      window.clearInterval(intervalRef.current)
      releaseStream()
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    },
    [],
  )

  const start = async () => {
    setError('')
    if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
      setError('Audio recording is not supported by this browser. Upload an audio clip instead.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data)
      }
      recorder.onerror = () => {
        setError('Recording failed. Please retry or upload an audio clip.')
        setState('idle')
        releaseStream()
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        urlRef.current = URL.createObjectURL(blob)
        setAudio({
          blob,
          url: urlRef.current,
          name: `recording-${Date.now()}.webm`,
          type: blob.type,
        })
        setState('ready')
        releaseStream()
      }
      recorder.start()
      setState('recording')
      setSeconds(0)
      intervalRef.current = window.setInterval(() => setSeconds((value) => value + 1), 1000)
    } catch (cause) {
      if (cause?.name === 'NotAllowedError' || cause?.name === 'SecurityError')
        setError(
          'Microphone permission was denied. Allow access in your browser settings, or upload an audio clip.',
        )
      else if (cause?.name === 'NotFoundError')
        setError('No microphone was found. Connect one or upload an audio clip.')
      else setError('The microphone could not be started. Please retry or upload an audio clip.')
      releaseStream()
    }
  }
  const stop = () => {
    window.clearInterval(intervalRef.current)
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }
  const selectUpload = (file) => {
    if (!file) {
      clearAudio()
      return
    }
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    urlRef.current = URL.createObjectURL(file)
    setAudio({ file, blob: file, url: urlRef.current, name: file.name, type: file.type })
  }

  return (
    <div className="recorder">
      {error && <Alert>{error}</Alert>}
      <div
        className={`recorder__stage ${state === 'recording' ? 'is-recording' : ''}`}
        aria-live="polite"
      >
        {state === 'idle' && (
          <>
            <button
              type="button"
              className="record-button"
              onClick={start}
              disabled={disabled}
              aria-label="Start recording"
            >
              <Icon name="mic" size={28} />
            </button>
            <strong>Ready to record</strong>
            <span>Speak clearly in Sinhala, or upload a clip below.</span>
          </>
        )}
        {state === 'recording' && (
          <>
            <div className="recording-pulse">
              <span />
            </div>
            <strong>
              Recording… <time>{time(seconds)}</time>
            </strong>
            <span>Your microphone is active</span>
            <button type="button" className="button button--danger" onClick={stop}>
              <Icon name="stop" size={17} /> Stop recording
            </button>
          </>
        )}
        {state === 'ready' && (
          <>
            <div className="audio-ready">
              <span>
                <Icon name="check" />
              </span>
              <div>
                <strong>Recording ready</strong>
                <small>{time(seconds)} · Review it before continuing</small>
              </div>
            </div>
            <audio controls src={audio.url} aria-label="Recorded audio preview" />
            <div className="button-row">
              <button type="button" className="button button--secondary" onClick={clearAudio}>
                <Icon name="trash" size={17} /> Discard & re-record
              </button>
              <button
                type="button"
                className="button button--primary"
                onClick={() => onUse(new File([audio.blob], audio.name, { type: audio.type }))}
                disabled={disabled}
              >
                <Icon name="check" size={17} /> Use recording
              </button>
            </div>
          </>
        )}
      </div>
      {state === 'idle' && (
        <>
          <p className="recorder__divider">or upload an audio clip</p>
          <FileUpload
            audioOnly
            id={`audio-${uploadId}`}
            file={audio?.file || null}
            onChange={selectUpload}
          />
          {audio?.file && (
            <div className="button-row button-row--end">
              <button
                type="button"
                className="button button--primary"
                disabled={disabled}
                onClick={() => onUse(audio.file)}
              >
                <Icon name="check" size={17} /> Use audio clip
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
