import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import Icon from '../components/Icon'
import { Alert, ConfirmDialog, PageHeader } from '../components/UI'

/**
 * Real per-student voice-command enrollment. Say each command a few
 * times so the app can recognize your voice saying it, not just the
 * words - this is what backs the embedding-matching half of voice
 * commands (app.streaming.embeddings on the backend); text-based
 * matching alone still works even without enrolling.
 *
 * Resumable by design: nothing here requires finishing all six commands
 * in one sitting. Reuses the same record -> MediaRecorder blob -> upload
 * capture as the dev sample-collection tool (VoiceSampleCollectorPage)
 * and AudioRecorder, rather than a second recorder implementation.
 */
export default function VoiceEnrollmentPage() {
  const [commands, setCommands] = useState(null)
  const [error, setError] = useState('')
  const [activeId, setActiveId] = useState(null)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [uploadingId, setUploadingId] = useState(null)
  const [feedback, setFeedback] = useState({}) // { [commandId]: { accepted, reason, similarity } }
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)
  const [deleting, setDeleting] = useState(false)

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  const load = () => {
    api
      .getVoiceEnrollmentStatus()
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
    setFeedback((prev) => ({ ...prev, [commandId]: null }))
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
        setUploadingId(commandId)
        try {
          const result = await api.submitVoiceEnrollmentSample(commandId, blob)
          setFeedback((prev) => ({ ...prev, [commandId]: result }))
          setCommands((prev) =>
            prev.map((command) =>
              command.id === commandId
                ? { ...command, collected: result.collected, complete: result.collected >= result.required }
                : command,
            ),
          )
        } catch (cause) {
          setError(cause.message)
        } finally {
          setUploadingId(null)
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

  const confirmDelete = async () => {
    const commandId = confirmDeleteId
    setDeleting(true)
    try {
      const result = await api.deleteVoiceEnrollmentSamples(commandId)
      setCommands((prev) =>
        prev.map((command) =>
          command.id === commandId
            ? { ...command, collected: result.collected, complete: false }
            : command,
        ),
      )
      setFeedback((prev) => ({ ...prev, [commandId]: null }))
    } catch (cause) {
      setError(cause.message)
    } finally {
      setDeleting(false)
      setConfirmDeleteId(null)
    }
  }

  const totalComplete = commands?.filter((command) => command.complete).length ?? 0

  return (
    <div className="page page--narrow">
      <PageHeader
        eyebrow="Speak to control the app"
        title="Voice command enrollment"
        description="Say each command a few times in your own voice. The app then recognizes it by how you say it, as a second check alongside recognizing the words themselves - useful if your speech is transcribed inconsistently. This is optional: voice commands already work from the words alone without enrolling."
      />
      {error && <Alert>{error}</Alert>}
      {commands === null ? (
        <p className="muted">Loading…</p>
      ) : (
        <>
          <div className="enrollment-summary">
            <div className="progress-track">
              <span style={{ width: `${(totalComplete / commands.length) * 100}%` }} />
            </div>
            <span>
              {totalComplete} of {commands.length} commands enrolled
            </span>
          </div>
          <div className="manage-list">
            {commands.map((command) => {
              const isActive = activeId === command.id
              const isRecording = isActive && recording
              const isUploading = uploadingId === command.id
              const result = feedback[command.id]
              return (
                <article key={command.id} className="enrollment-command">
                  <div>
                    <div className="document-icon">
                      <Icon name="mic" />
                    </div>
                    <div>
                      <strong>
                        {command.phrase}
                        {command.destructive && <span className="badge">Destructive</span>}
                      </strong>
                      <small>
                        {command.collected} of {command.required} samples
                        {command.complete && ' · complete'}
                      </small>
                      {result && (
                        <p
                          className={`enrollment-feedback ${result.accepted ? 'is-accepted' : 'is-rejected'}`}
                        >
                          <Icon name={result.accepted ? 'check' : 'alert'} size={14} />
                          {result.accepted
                            ? 'Accepted — sounded consistent with your other takes.'
                            : result.reason === 'low_similarity'
                              ? "Didn't sound like your earlier takes of this command. Try saying it again, clearly."
                              : 'This command already has enough samples.'}
                          {result.similarity != null && ` (${Math.round(result.similarity * 100)}% match)`}
                        </p>
                      )}
                    </div>
                  </div>
                  {isRecording ? (
                    <button type="button" className="button button--danger button--small" onClick={stop}>
                      <Icon name="stop" size={15} /> Stop ({seconds}s)
                    </button>
                  ) : (
                    <div className="row-actions">
                      {command.collected > 0 && (
                        <button
                          type="button"
                          className="icon-button icon-button--danger"
                          title={`Delete all ${command.id} samples`}
                          onClick={() => setConfirmDeleteId(command.id)}
                          disabled={recording || isUploading}
                        >
                          <Icon name="trash" size={16} />
                        </button>
                      )}
                      <button
                        type="button"
                        className="button button--secondary button--small"
                        disabled={recording || isUploading || command.complete}
                        onClick={() => start(command.id)}
                      >
                        {isUploading ? (
                          <span className="spinner spinner--small" />
                        ) : (
                          <Icon name="mic" size={15} />
                        )}{' '}
                        {command.complete ? 'Complete' : 'Record'}
                      </button>
                    </div>
                  )}
                </article>
              )
            })}
          </div>
        </>
      )}
      <ConfirmDialog
        open={confirmDeleteId !== null}
        title="Delete these samples?"
        message="All recorded samples for this command will be removed. You can re-record them any time."
        confirmLabel="Delete samples"
        dangerous
        busy={deleting}
        onCancel={() => setConfirmDeleteId(null)}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
