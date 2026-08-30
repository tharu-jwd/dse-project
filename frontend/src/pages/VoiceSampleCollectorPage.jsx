import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import Icon from '../components/Icon'
import { Alert, PageHeader } from '../components/UI'

/**
 * Dev-only tool: step 1 of the voice-command embedding work needs real
 * recordings of the six command phrases to answer "do same-phrase
 * embeddings cluster?" - this records them straight from the browser
 * instead of requiring a manual file transfer. Not the real per-student
 * enrollment UI (that depends on what this data shows).
 */
export default function VoiceSampleCollectorPage() {
  const [commands, setCommands] = useState(null)
  const [error, setError] = useState('')
  const [activeId, setActiveId] = useState(null)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  const load = () => {
    api
      .getVoiceSampleProgress()
      .then((data) => setCommands(data.commands))
      .catch((cause) => setError(cause.message))
  }
  useEffect(load, [])
  useEffect(
    () => () => {
      window.clearInterval(intervalRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
    },
    [],
  )

  const start = async (commandId) => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data)
      }
      recorder.onstop = async () => {
        streamRef.current?.getTracks().forEach((track) => track.stop())
        window.clearInterval(intervalRef.current)
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        setRecording(false)
        try {
          const result = await api.uploadVoiceSample(commandId, blob)
          setCommands((prev) =>
            prev.map((command) =>
              command.id === commandId ? { ...command, count: result.counts[commandId] } : command,
            ),
          )
        } catch (cause) {
          setError(cause.message)
        }
      }
      recorder.start()
      setActiveId(commandId)
      setRecording(true)
      setSeconds(0)
      intervalRef.current = window.setInterval(() => setSeconds((value) => value + 1), 1000)
    } catch {
      setError('Microphone permission was denied, or no microphone is available.')
    }
  }

  const stop = () => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }

  const reRecord = async (commandId) => {
    setError('')
    try {
      const result = await api.deleteVoiceSamples(commandId)
      setCommands((prev) =>
        prev.map((command) =>
          command.id === commandId ? { ...command, count: result.counts[commandId] } : command,
        ),
      )
    } catch (cause) {
      setError(cause.message)
    }
  }

  return (
    <div className="page page--narrow">
      <PageHeader
        eyebrow="Dev tool · not the real enrollment UI"
        title="Collect voice-command samples"
        description="Say each phrase 3-5 times, in a few different sessions if you can. These get saved as {command}_{n}.wav on the server for scripts/validate_command_embeddings.py."
      />
      {error && <Alert>{error}</Alert>}
      {commands === null ? (
        <p className="muted">Loading…</p>
      ) : (
        <div className="manage-list">
          {commands.map((command) => (
            <article key={command.id}>
              <div>
                <div className="document-icon">
                  <Icon name="mic" />
                </div>
                <div>
                  <strong>{command.phrase}</strong>
                  <small>
                    {command.id} · {command.count} sample{command.count === 1 ? '' : 's'}
                  </small>
                </div>
              </div>
              {activeId === command.id && recording ? (
                <button type="button" className="button button--danger button--small" onClick={stop}>
                  <Icon name="stop" size={15} /> Stop ({seconds}s)
                </button>
              ) : (
                <div className="row-actions">
                  {command.count > 0 && (
                    <button
                      type="button"
                      className="icon-button icon-button--danger"
                      title={`Delete all ${command.id} samples`}
                      onClick={() => reRecord(command.id)}
                    >
                      <Icon name="trash" size={16} />
                    </button>
                  )}
                  <button
                    type="button"
                    className="button button--secondary button--small"
                    disabled={recording}
                    onClick={() => start(command.id)}
                  >
                    <Icon name="mic" size={15} /> Record
                  </button>
                </div>
              )}
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
