import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import Icon from '../components/Icon'
import { Alert, ConfirmDialog, PageHeader } from '../components/UI'
import { useLanguage } from '../contexts/LanguageContext'

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
  const { t } = useLanguage()
  const LANGUAGE_LABELS = { si: t('lang.sinhala'), en: t('lang.english') }
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
  // A recorded-but-not-yet-submitted enrollment take: { commandId, blob, url }.
  // Nothing is uploaded (and no embedding created) until the student
  // listens back and explicitly submits it.
  const [pendingSample, setPendingSample] = useState(null)
  const pendingUrlRef = useRef('')

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
      if (pendingUrlRef.current) URL.revokeObjectURL(pendingUrlRef.current)
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
    if (pendingUrlRef.current) URL.revokeObjectURL(pendingUrlRef.current)
    setPendingSample(null)
    try {
      await startRecorder(async (blob) => {
        setRecording(false)
        // Hold the take locally for review instead of uploading it right
        // away - nothing is submitted (and no embedding created) until
        // the student has listened back and confirmed it.
        const url = URL.createObjectURL(blob)
        pendingUrlRef.current = url
        setPendingSample({ commandId, blob, url })
      })
      setActiveId(commandId)
      setRecording(true)
    } catch {
      setError(t('enroll.micDenied'))
    }
  }

  const stop = () => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }

  const discardSample = () => {
    if (pendingUrlRef.current) URL.revokeObjectURL(pendingUrlRef.current)
    pendingUrlRef.current = ''
    setPendingSample(null)
  }

  const submitSample = async () => {
    if (!pendingSample) return
    const { commandId, blob, url } = pendingSample
    setUploadingId(commandId)
    setError('')
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
      URL.revokeObjectURL(url)
      pendingUrlRef.current = ''
      setPendingSample(null)
    } catch (cause) {
      setError(cause.message)
    } finally {
      setUploadingId(null)
    }
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
      setError(t('enroll.micDenied'))
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
        eyebrow={t('enroll.eyebrow')}
        title={t('enroll.title')}
        description={t('enroll.description')}
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
              <small>{activeLanguage === lng ? t('enroll.currentlyActive') : t('enroll.notActive')}</small>
            </span>
          </button>
        ))}
      </div>

      {activeLanguage !== null && activeLanguage !== language && (
        <Alert type="info">
          {t('enroll.settingUpButActive', LANGUAGE_LABELS[language], LANGUAGE_LABELS[activeLanguage])}{' '}
          <button
            type="button"
            className="button button--secondary button--small"
            onClick={makeActive}
            disabled={switchingLanguage}
            style={{ marginLeft: 8 }}
          >
            {switchingLanguage ? t('enroll.switching') : t('enroll.makeActive', LANGUAGE_LABELS[language])}
          </button>
        </Alert>
      )}
      {activeLanguage !== null && activeLanguage === language && (
        <Alert type="success">{t('enroll.activeForLiveCommands', LANGUAGE_LABELS[language])}</Alert>
      )}

      {error && <Alert>{error}</Alert>}
      {commands === null ? (
        <p className="muted">{t('enroll.loading')}</p>
      ) : (
        <>
          <div className="enrollment-summary">
            <div className="progress-track">
              <span style={{ width: `${(totalComplete / commands.length) * 100}%` }} />
            </div>
            <span>{t('enroll.commandsEnrolled', totalComplete, commands.length)}</span>
          </div>
          <div className="manage-list">
            {commands.map((command) => {
              const isActive = activeId === command.id
              const isRecording = isActive && recording
              const isUploading = uploadingId === command.id
              const isPendingReview = pendingSample?.commandId === command.id
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
                        {command.destructive && <span className="badge">{t('enroll.destructive')}</span>}
                      </strong>
                      <small>
                        {t('enroll.samplesOf', command.collected, command.required)}
                        {command.complete && t('enroll.complete')}
                      </small>
                      {result && (
                        <p
                          className={`enrollment-feedback ${result.accepted ? 'is-accepted' : 'is-rejected'}`}
                        >
                          <Icon name={result.accepted ? 'check' : 'alert'} size={14} />
                          {result.accepted
                            ? t('enroll.accepted')
                            : result.reason === 'low_similarity'
                              ? t('enroll.lowSimilarity')
                              : t('enroll.alreadyEnough')}
                          {result.similarity != null && t('enroll.matchPercent', Math.round(result.similarity * 100))}
                        </p>
                      )}
                      {isPendingReview && (
                        <div className="enrollment-review">
                          <audio controls src={pendingSample.url} aria-label={t('enroll.reviewAudioLabel')} />
                          <p className="muted" style={{ fontSize: '0.78rem', margin: '4px 0 0' }}>
                            {t('enroll.reviewHint')}
                          </p>
                        </div>
                      )}
                      {practice && (
                        <p
                          className={`enrollment-feedback ${practice.passesThreshold ? 'is-accepted' : 'is-rejected'}`}
                        >
                          <Icon name={practice.passesThreshold ? 'check' : 'alert'} size={14} />
                          {practice.ownSimilarity == null
                            ? t('enroll.noEnrolledSamples')
                            : practice.passesThreshold
                              ? t('enroll.practiceMatch', Math.round(practice.ownSimilarity * 100))
                              : t('enroll.practiceNoMatch', Math.round(practice.ownSimilarity * 100))}
                          {practice.closestOtherCommandId && (
                            <>
                              {t(
                                'enroll.closerToOther',
                                practice.closestOtherCommandId,
                                Math.round(practice.closestOtherSimilarity * 100),
                              )}
                            </>
                          )}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="row-actions">
                    {isRecording ? (
                      <button type="button" className="button button--danger button--small" onClick={stop}>
                        <Icon name="stop" size={15} /> {t('enroll.stopSeconds', seconds)}
                      </button>
                    ) : isPendingReview ? (
                      <>
                        <button
                          type="button"
                          className="button button--secondary button--small"
                          onClick={discardSample}
                          disabled={isUploading}
                        >
                          <Icon name="mic" size={15} /> {t('enroll.reRecord')}
                        </button>
                        <button
                          type="button"
                          className="button button--primary button--small"
                          onClick={submitSample}
                          disabled={isUploading}
                        >
                          {isUploading ? (
                            <span className="spinner spinner--small" />
                          ) : (
                            <Icon name="check" size={15} />
                          )}{' '}
                          {t('enroll.submitSample')}
                        </button>
                      </>
                    ) : (
                      <>
                        {command.collected > 0 && (
                          <button
                            type="button"
                            className="icon-button icon-button--danger"
                            title={t('enroll.deleteAllSamples', command.id)}
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
                          {command.complete ? t('enroll.completeLabel') : t('enroll.record')}
                        </button>
                      </>
                    )}
                    {!isPendingReview &&
                      command.collected > 0 &&
                      (isPracticeRecording ? (
                        <button
                          type="button"
                          className="button button--danger button--small"
                          onClick={stopPractice}
                        >
                          <Icon name="stop" size={15} /> {t('enroll.stopSeconds', seconds)}
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="button button--text button--small"
                          disabled={practiceRecording || isPracticeUploading}
                          onClick={() => startPractice(command.id)}
                          title={t('enroll.tryMatchTitle')}
                        >
                          {isPracticeUploading ? (
                            <span className="spinner spinner--small" />
                          ) : (
                            <Icon name="quiz" size={15} />
                          )}{' '}
                          {t('enroll.practice')}
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
        title={t('enroll.deleteSamplesTitle')}
        message={t('enroll.deleteSamplesMessage')}
        confirmLabel={t('enroll.deleteSamples')}
        dangerous
        busy={deleting}
        onCancel={() => setConfirmDeleteId(null)}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
