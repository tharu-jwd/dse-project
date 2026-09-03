import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import lectureBackground from '../assets/2.jpg'
import noteBackground from '../assets/6.jpg'
import rocketImage from '../assets/rocket.png'
import AudioRecorder from '../components/AudioRecorder'
import FileUpload, { validateMediaFile } from '../components/FileUpload'
import Icon from '../components/Icon'
import LiveTranscription from '../components/LiveTranscription'
import TranscriptEditor from '../components/TranscriptEditor'
import TranscriptionStatus from '../components/TranscriptionStatus'
import useTranscriptionJob from '../components/useTranscriptionJob'
import { Loading, PageHeader } from '../components/UI'
import { useLanguage } from '../contexts/LanguageContext'

export default function CreateTranscriptPage({ type = 'LECTURE' }) {
  const { t, language } = useLanguage()
  const isNote = type === 'NOTE'
  const [title, setTitle] = useState('')
  const [file, setFile] = useState(null)
  const [errors, setErrors] = useState({})
  const [inputMode, setInputMode] = useState('live')
  const [recent, setRecent] = useState(null)
  const [liveResult, setLiveResult] = useState(null)
  const [liveLoading, setLiveLoading] = useState(false)
  const { job, start, reset } = useTranscriptionJob()
  const navigate = useNavigate()
  const handleLiveSessionEnd = (transcriptId) => {
    setLiveLoading(true)
    api
      .getTranscript(transcriptId)
      .then(setLiveResult)
      .catch(() => navigate(`/transcripts/${transcriptId}`))
      .finally(() => setLiveLoading(false))
  }
  useEffect(() => {
    if (job?.status === 'COMPLETED' && job.transcriptId) {
      const timer = window.setTimeout(() => navigate(`/transcripts/${job.transcriptId}`), 550)
      return () => window.clearTimeout(timer)
    }
  }, [job, navigate])
  useEffect(() => {
    let active = true
    api
      .getTranscripts()
      .then((items) => active && setRecent(items.slice(0, 3)))
      .catch(() => active && setRecent([]))
    return () => {
      active = false
    }
  }, [])
  const begin = (mediaFile = file) => {
    const next = {}
    if (!title.trim())
      next.title = t(isNote ? 'create.noteTitleRequired' : 'create.lectureTitleRequired')
    const fileError = validateMediaFile(mediaFile, { audioOnly: isNote, language })
    if (fileError) next.file = fileError
    setErrors(next)
    if (Object.keys(next).length) return
    setFile(mediaFile)
    const data = new FormData()
    data.append('file', mediaFile)
    data.append('title', title.trim())
    data.append('type', type)
    start(data)
  }
  if (job)
    return (
      <div
        className="page page--narrow has-bg-image"
        style={{ backgroundImage: `url(${isNote ? noteBackground : lectureBackground})` }}
      >
        <PageHeader
          eyebrow={t(isNote ? 'nav.selfStudyNotes' : 'nav.lectureCaptioning')}
          title={
            job.status === 'FAILED' ? t('create.somethingWentWrong') : t('create.creatingTranscript')
          }
          description={title}
        />
        <TranscriptionStatus
          status={job.status}
          message={job.message}
          onRetry={() => {
            reset()
            window.setTimeout(() => begin(file), 0)
          }}
        />
      </div>
    )
  if (liveLoading)
    return (
      <div
        className="page page--narrow has-bg-image"
        style={{ backgroundImage: `url(${noteBackground})` }}
      >
        <Loading label={t('create.preparingNote')} />
      </div>
    )
  if (liveResult)
    return (
      <div
        className="page page--editor has-bg-image"
        style={{ backgroundImage: `url(${noteBackground})` }}
      >
        <PageHeader
          eyebrow={t('nav.selfStudyNotes')}
          title={t('create.reviewYourNote')}
          description={t('create.reviewNoteDescription')}
          back={
            <button className="back-link" onClick={() => setLiveResult(null)}>
              {t('create.recordAnotherNote')}
            </button>
          }
        />
        <TranscriptEditor initialTranscript={liveResult} onTranscriptChange={setLiveResult} />
      </div>
    )
  return (
    <div
      className="page has-bg-image"
      style={{ backgroundImage: `url(${isNote ? noteBackground : lectureBackground})` }}
    >
      <div className="upload-hero">
        <span className="eyebrow">
          {t(isNote ? 'create.speakSaveStudy' : 'create.accessibleLearningStart')}
        </span>
        <h1>{t(isNote ? 'create.createSelfStudyNote' : 'create.captionRecordedLecture')}</h1>
        <p className="muted">
          {t(isNote ? 'create.noteHelperText' : 'create.lectureHelperText')}
        </p>
      </div>
      <div className={`upload-grid${isNote ? ' upload-grid--note' : ''}`}>
        {isNote ? (
          <section className="glass-card">
            <div className="form-card__heading">
              <span>1</span>
              <div>
                <h2>{t('create.nameYourNote')}</h2>
                <p>{t('create.nameYourNoteHelp')}</p>
              </div>
            </div>
            <div className="field">
              <label htmlFor="media-title">{t('create.noteTitleLabel')}</label>
              <input
                id="media-title"
                value={title}
                onChange={(e) => {
                  setTitle(e.target.value)
                  setErrors({ ...errors, title: '' })
                }}
                placeholder={t('create.noteTitlePlaceholder')}
                aria-invalid={Boolean(errors.title)}
              />
              {errors.title && (
                <span className="field-error" role="alert">
                  {errors.title}
                </span>
              )}
            </div>
            <div className="form-card__heading" style={{ marginTop: 28 }}>
              <span>2</span>
              <div>
                <h2>{t('create.addVoiceNote')}</h2>
                <p>{t('create.addVoiceNoteHelp')}</p>
              </div>
            </div>
            <div className="tabs" role="tablist" aria-label={t('create.noteInputMode')}>
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === 'capture'}
                className={inputMode === 'capture' ? 'active' : ''}
                onClick={() => setInputMode('capture')}
              >
                <Icon name="mic" size={18} /> {t('create.recordOrUpload')}
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === 'live'}
                className={inputMode === 'live' ? 'active' : ''}
                onClick={() => setInputMode('live')}
              >
                <Icon name="mic" size={18} /> {t('create.liveTranscription')}
              </button>
            </div>
            {inputMode === 'capture' ? (
              <AudioRecorder onUse={begin} />
            ) : (
              <LiveTranscription title={title} onSessionEnd={handleLiveSessionEnd} />
            )}
          </section>
        ) : (
          <section className="glass-card upload-tile">
            <div className="field">
              <label htmlFor="media-title">{t('create.lectureTitleLabel')}</label>
              <input
                id="media-title"
                value={title}
                onChange={(e) => {
                  setTitle(e.target.value)
                  setErrors({ ...errors, title: '' })
                }}
                placeholder={t('create.lectureTitlePlaceholder')}
                aria-invalid={Boolean(errors.title)}
              />
              {errors.title && (
                <span className="field-error" role="alert">
                  {errors.title}
                </span>
              )}
            </div>
            <FileUpload
              file={file}
              onChange={(value) => {
                setFile(value)
                setErrors({ ...errors, file: '' })
              }}
              error={errors.file}
              iconImage={rocketImage}
              floatingIcon
              heading={t('create.readyForLiftoff')}
              tagline={t('create.dragDropLecture')}
            />
            <div className="button-row button-row--end">
              <button type="button" className="button button--primary" onClick={() => begin()}>
                <Icon name="upload" size={17} /> {t('create.uploadTranscribe')}
              </button>
            </div>
          </section>
        )}
        <div className="upload-grid__side">
          <div className="glass-card glass-card--tight">
            <h3>{t(isNote ? 'create.recentUploads' : 'create.recentTranscriptions')}</h3>
            {recent === null ? (
              <Loading label={t('ui.loading')} />
            ) : recent.length ? (
              <div className="recent-list">
                {recent.map((item) => (
                  <Link to={`/transcripts/${item.id}`} key={item.id} className="recent-item">
                    <span
                      className={`document-icon document-icon--${item.type.toLowerCase()}`}
                    >
                      <Icon
                        name={
                          item.type === 'NOTE'
                            ? 'mic'
                            : item.type === 'QUIZ_ANSWER'
                              ? 'quiz'
                              : 'file'
                        }
                      />
                    </span>
                    <span>
                      <strong>{item.title}</strong>
                      <small>
                        {new Intl.DateTimeFormat(language === 'si' ? 'si-LK' : 'en-GB', {
                          day: 'numeric',
                          month: 'short',
                        }).format(new Date(item.date))}
                      </small>
                    </span>
                  </Link>
                ))}
              </div>
            ) : (
              <p className="muted" style={{ fontSize: '0.8rem' }}>
                {t('create.nothingHereYet')}
              </p>
            )}
          </div>
          {isNote && (
            <div className="glass-card glass-card--tight">
              <h3>{t('create.privacyTitle')}</h3>
              <p className="muted" style={{ fontSize: '0.8rem', lineHeight: 1.6 }}>
                {t('create.privacyBody')}
              </p>
            </div>
          )}
        </div>
      </div>
      <div className="feature-row">
        <div>
          <span className="feature-row__icon">
            <Icon name="check" size={24} />
          </span>
          <h5>{t('create.featureAccuracyTitle')}</h5>
          <p className="muted">{t('create.featureAccuracyBody')}</p>
        </div>
        <div>
          <span className="feature-row__icon">
            <Icon name="users" size={24} />
          </span>
          <h5>{t('create.featureClassroomsTitle')}</h5>
          <p className="muted">{t('create.featureClassroomsBody')}</p>
        </div>
        <div>
          <span className="feature-row__icon">
            <Icon name="settings" size={24} />
          </span>
          <h5>{t('create.featureInclusiveTitle')}</h5>
          <p className="muted">{t('create.featureInclusiveBody')}</p>
        </div>
      </div>
    </div>
  )
}
