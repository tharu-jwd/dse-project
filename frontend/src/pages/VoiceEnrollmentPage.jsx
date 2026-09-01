import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import Icon from '../components/Icon'
import { Alert, ConfirmDialog, PageHeader } from '../components/UI'

const LANGUAGE_LABELS = { si: 'Sinhala', en: 'English' }

/**
 * Real per-student voice-command setup. Say each command a few times so
 * the app can recognize your voice saying it, not just the words - this
 * is what backs the embedding-matching half of voice commands
 * (app.streaming.embeddings on the backend); text-based matching alone
 * still works even without enrolling.
 *
 * Covers both phrase sets (Sinhala/English) via the `language` query
 * param - switching which one is active for live commands is a database
 * setting (User.command_language), never a code change, and deleting +
 * re-recording a command's samples is a pure data operation too: the
 * backend re-embeds automatically on every upload.
 *
 * Resumable by design: nothing here requires finishing all six commands
 * in one sitting. Reuses the same record -> MediaRecorder blob -> upload
 * capture as the dev sample-collection tool (VoiceSampleCollectorPage)
 * and AudioRecorder, rather than a second recorder implementation.
 */
export default function VoiceEnrollmentPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const language = searchParams.get('language') === 'en' ? 'en' : 'si'

  const [commands, setCommands] = useState(null)
  const [activeLanguage, setActiveLanguageState] = useState(null)
  const [switchingLanguage, setSwitchingLanguage] = useState(false)
  const [error, setError] = useState('')
  const [activeId, setActiveId] = useState(null)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [uploadingId, setUploadingId] = useState(null)
  const [feedback, setFeedback] = useState({}) // { [commandId]: { accepted, reason, similarity } }
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)
  const [deleting, setDeleting] = useState(false)

  // Practice mode: a separate recorder from enrollment, keyed by command id.
  const [practiceId, setPracticeId] = useState(null)
  const [practiceRecording, setPracticeRecording] = useState(false)
  const [practiceUploadingId, setPracticeUploadingId] = useState(null)
  const [practiceResult, setPracticeResult] = useState({}) // { [commandId]: result }

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  const load = () => {
    setCommands(null)
    api
      .getVoiceEnrollmentStatus(language)
      .then((data) => {
        setCommands(data.commands)
        setActiveLanguageState(data.activeLanguage)
      })
      .catch((cause) => setError(cause.message))
  }
  useEffect(load, [language])
  useEffect(
    () => () => {
      window.clearInterval(intervalRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
    },
    [],
  )

  const startRecorder = async (onStop) => {
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
      await onStop(blob)
    }
    recorder.start()
    setSeconds(0)
    intervalRef.current = window.setInterval(() => setSeconds((value) => value + 1), 1000)
  }

  const start = async (commandId) => {
    setError('')
    setFeedback((prev) => ({ ...prev, [commandId]: null }))
    try {
      await startRecorder(async (blob) => {
        setRecording(false)
        setUploadingId(commandId)
        try {
          const result = await api.submitVoiceEnrollmentSample(commandId, blob, language)
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
      })
      setActiveId(commandId)
      setRecording(true)
    } catch {
      setError('Microphone permission was denied, or no microphone is available.')
    }
  }

  const stop = () => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }

  const startPractice = async (commandId) => {
    setError('')
    setPracticeResult((prev) => ({ ...prev, [commandId]: null }))
    try {
      await startRecorder(async (blob) => {
        setPracticeRecording(false)
        setPracticeUploadingId(commandId)
        try {
          const result = await api.practiceVoiceCommand(commandId, blob, language)
          setPracticeResult((prev) => ({ ...prev, [commandId]: result }))
        } catch (cause) {
          setError(cause.message)
        } finally {
          setPracticeUploadingId(null)
        }
      })
      setPracticeId(commandId)
      setPracticeRecording(true)
    } catch {
      setError('Microphone permission was denied, or no microphone is available.')
    }
  }

  const stopPractice = () => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }

  const confirmDelete = async () => {
    const commandId = confirmDeleteId
    setDeleting(true)
    try {
      const result = await api.deleteVoiceEnrollmentSamples(commandId, language)
      setCommands((prev) =>
        prev.map((command) =>
          command.id === commandId
            ? { ...command, collected: result.collected, complete: false }
            : command,
        ),
      )
      setFeedback((prev) => ({ ...prev, [commandId]: null }))
      setPracticeResult((prev) => ({ ...prev, [commandId]: null }))
    } catch (cause) {
      setError(cause.message)
    } finally {
      setDeleting(false)
      setConfirmDeleteId(null)
    }
  }

  const makeActive = async () => {
    setSwitchingLanguage(true)
    setError('')
    try {
      const result = await api.setActiveCommandLanguage(language)
      setActiveLanguageState(result.activeLanguage)
    } catch (cause) {
      setError(cause.message)
    } finally {
      setSwitchingLanguage(false)
    }
  }

  const totalComplete = commands?.filter((command) => command.complete).length ?? 0

  return (
    <div className="page page--narrow">
      <PageHeader
        eyebrow="Speak to control the app"
        title="Voice command setup"
        description="Say each command a few times in your own voice. The app then recognizes it by how you say it, as a second check alongside recognizing the words themselves - useful if your speech is transcribed inconsistently. This is optional: voice commands already work from the words alone without enrolling."
      />

      <div className="voice-setup-options" style={{ marginBottom: 20 }}>
        {(['si', 'en']).map((lng) => (
          <button
            key={lng}
            type="button"
            className="voice-setup-option"
            aria-current={language === lng}
            style={
              language === lng
                ? { borderColor: 'var(--teal)', background: 'var(--teal-soft)' }
                : undefined
            }
            onClick={() => setSearchParams({ language: lng })}
          >
            <span className="voice-setup-option__icon">
              <Icon name="mic" size={17} />
            </span>
            <span>
              <strong>{LANGUAGE_LABELS[lng]}</strong>
              <small>{activeLanguage === lng ? 'Currently active for live commands' : 'Not active'}</small>
            </span>
          </button>
        ))}
      </div>

      {activeLanguage !== null && activeLanguage !== language && (
        <Alert type="info">
          You're setting up {LANGUAGE_LABELS[language]} commands, but {LANGUAGE_LABELS[activeLanguage]} is
          currently active for live voice commands.{' '}
          <button
            type="button"
            className="button button--secondary button--small"
            onClick={makeActive}
            disabled={switchingLanguage}
            style={{ marginLeft: 8 }}
          >
            {switchingLanguage ? 'Switching…' : `Make ${LANGUAGE_LABELS[language]} active`}
          </button>
        </Alert>
      )}
      {activeLanguage !== null && activeLanguage === language && (
        <Alert type="success">{LANGUAGE_LABELS[language]} is active for your live voice commands.</Alert>
      )}

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

              const isPracticeActive = practiceId === command.id
              const isPracticeRecording = isPracticeActive && practiceRecording
              const isPracticeUploading = practiceUploadingId === command.id
              const practice = practiceResult[command.id]

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
                      {practice && (
                        <p
                          className={`enrollment-feedback ${practice.passesThreshold ? 'is-accepted' : 'is-rejected'}`}
                        >
                          <Icon name={practice.passesThreshold ? 'check' : 'alert'} size={14} />
                          {practice.ownSimilarity == null
                            ? 'No enrolled samples to compare against yet.'
                            : practice.passesThreshold
                              ? `Practice: sounds like you — ${Math.round(practice.ownSimilarity * 100)}% match.`
                              : `Practice: didn't quite match — ${Math.round(practice.ownSimilarity * 100)}% match.`}
                          {practice.closestOtherCommandId && (
                            <>
                              {' '}
                              (closer to "{practice.closestOtherCommandId}" —{' '}
                              {Math.round(practice.closestOtherSimilarity * 100)}% — say it more clearly)
                            </>
                          )}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="row-actions">
                    {isRecording ? (
                      <button type="button" className="button button--danger button--small" onClick={stop}>
                        <Icon name="stop" size={15} /> Stop ({seconds}s)
                      </button>
                    ) : (
                      <>
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
                      </>
                    )}
                    {command.collected > 0 &&
                      (isPracticeRecording ? (
                        <button
                          type="button"
                          className="button button--danger button--small"
                          onClick={stopPractice}
                        >
                          <Icon name="stop" size={15} /> Stop ({seconds}s)
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="button button--text button--small"
                          disabled={practiceRecording || isPracticeUploading}
                          onClick={() => startPractice(command.id)}
                          title="Try saying the command and see how well it matches your enrolled samples"
                        >
                          {isPracticeUploading ? (
                            <span className="spinner spinner--small" />
                          ) : (
                            <Icon name="quiz" size={15} />
                          )}{' '}
                          Practice
                        </button>
                      ))}
                  </div>
                </article>
              )
            })}
          </div>
        </>
      )}
      <ConfirmDialog
        open={confirmDeleteId !== null}
        title="Delete these samples?"
        message="All recorded samples for this command, in this language, will be removed. You can re-record them any time - re-recording always creates fresh embeddings automatically."
        confirmLabel="Delete samples"
        dangerous
        busy={deleting}
        onCancel={() => setConfirmDeleteId(null)}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
