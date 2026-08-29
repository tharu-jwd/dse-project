import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import lectureBackground from '../assets/2.jpg'
import rocketImage from '../assets/rocket.png'
import AudioRecorder from '../components/AudioRecorder'
import FileUpload, { validateMediaFile } from '../components/FileUpload'
import Icon from '../components/Icon'
import LiveTranscription from '../components/LiveTranscription'
import TranscriptionStatus from '../components/TranscriptionStatus'
import useTranscriptionJob from '../components/useTranscriptionJob'
import { Loading, PageHeader } from '../components/UI'

export default function CreateTranscriptPage({ type = 'LECTURE' }) {
  const isNote = type === 'NOTE'
  const [title, setTitle] = useState('')
  const [file, setFile] = useState(null)
  const [errors, setErrors] = useState({})
  const [inputMode, setInputMode] = useState('capture')
  const [recent, setRecent] = useState(null)
  const { job, start, reset } = useTranscriptionJob()
  const navigate = useNavigate()
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
    if (!title.trim()) next.title = `${isNote ? 'Note' : 'Lecture'} title is required.`
    const fileError = validateMediaFile(mediaFile, { audioOnly: isNote })
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
        className={`page page--narrow ${isNote ? '' : 'has-bg-image'}`}
        style={isNote ? undefined : { backgroundImage: `url(${lectureBackground})` }}
      >
        <PageHeader
          eyebrow={isNote ? 'Self-study notes' : 'Lecture captioning'}
          title={job.status === 'FAILED' ? 'Something went wrong' : 'Creating your transcript'}
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
  return (
    <div
      className={`page ${isNote ? '' : 'has-bg-image'}`}
      style={isNote ? undefined : { backgroundImage: `url(${lectureBackground})` }}
    >
      <div className="upload-hero">
        <span className="eyebrow">
          {isNote ? 'Speak it. Save it. Study it.' : 'Accessible learning starts here'}
        </span>
        <h1>{isNote ? 'Create a self-study note' : 'Caption a recorded lecture'}</h1>
        <p className="muted">
          {isNote
            ? 'Record a thought or upload an audio clip. We’ll turn it into editable Sinhala text.'
            : 'Upload an audio or video recording and SinhaSpeech will create an editable Sinhala transcript.'}
        </p>
      </div>
      <div className="upload-grid">
        {isNote ? (
          <section className="glass-card">
            <div className="form-card__heading">
              <span>1</span>
              <div>
                <h2>Name your note</h2>
                <p>Use a clear title so you can find it later.</p>
              </div>
            </div>
            <div className="field">
              <label htmlFor="media-title">Note title</label>
              <input
                id="media-title"
                value={title}
                onChange={(e) => {
                  setTitle(e.target.value)
                  setErrors({ ...errors, title: '' })
                }}
                placeholder="e.g. Database revision — Week 4"
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
                <h2>Add your voice note</h2>
                <p>Record here or choose an audio clip you already have.</p>
              </div>
            </div>
            <div className="tabs" role="tablist" aria-label="Note input mode">
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === 'capture'}
                className={inputMode === 'capture' ? 'active' : ''}
                onClick={() => setInputMode('capture')}
              >
                <Icon name="mic" size={18} /> Record or upload
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={inputMode === 'live'}
                className={inputMode === 'live' ? 'active' : ''}
                onClick={() => setInputMode('live')}
              >
                <Icon name="mic" size={18} /> Live transcription
              </button>
            </div>
            {inputMode === 'capture' ? (
              <AudioRecorder onUse={begin} />
            ) : (
              <LiveTranscription
                title={title}
                onSessionEnd={(transcriptId) => navigate(`/transcripts/${transcriptId}`)}
              />
            )}
          </section>
        ) : (
          <section className="glass-card upload-tile">
            <div className="field">
              <label htmlFor="media-title">Lecture title</label>
              <input
                id="media-title"
                value={title}
                onChange={(e) => {
                  setTitle(e.target.value)
                  setErrors({ ...errors, title: '' })
                }}
                placeholder="e.g. Introduction to Algorithms — Week 2"
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
              heading="Ready for liftoff?"
              tagline="Drag and drop your lecture files here or use the button below."
            />
            <div className="button-row button-row--end">
              <button type="button" className="button button--primary" onClick={() => begin()}>
                <Icon name="upload" size={17} /> Upload &amp; transcribe
              </button>
            </div>
          </section>
        )}
        <div className="upload-grid__side">
          <div className="glass-card glass-card--tight">
            <h3>{isNote ? 'Recent uploads' : 'Recent transcriptions'}</h3>
            {recent === null ? (
              <Loading label="Loading…" />
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
                        {new Intl.DateTimeFormat('en-GB', {
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
                Nothing here yet.
              </p>
            )}
          </div>
          {isNote && (
            <div className="glass-card glass-card--tight">
              <h3>Your privacy matters</h3>
              <p className="muted" style={{ fontSize: '0.8rem', lineHeight: 1.6 }}>
                Recordings are stored securely for transcription and later review. Only you and
                authorized course staff can access them. Do not upload sensitive personal
                information.
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
          <h5>Sinhala-tuned accuracy</h5>
          <p className="muted">
            Powered by a fine-tuned Whisper model trained specifically for Sinhala speech.
          </p>
        </div>
        <div>
          <span className="feature-row__icon">
            <Icon name="users" size={24} />
          </span>
          <h5>Built for classrooms</h5>
          <p className="muted">
            Students and teachers each get workflows suited to captioning, notes and review.
          </p>
        </div>
        <div>
          <span className="feature-row__icon">
            <Icon name="settings" size={24} />
          </span>
          <h5>Inclusive by design</h5>
          <p className="muted">
            Adjustable text size, high contrast and confidence flags support every learner.
          </p>
        </div>
      </div>
    </div>
  )
}
