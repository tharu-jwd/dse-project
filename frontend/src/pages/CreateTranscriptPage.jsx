import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import AudioRecorder from '../components/AudioRecorder'
import FileUpload, { validateMediaFile } from '../components/FileUpload'
import Icon from '../components/Icon'
import LiveTranscription from '../components/LiveTranscription'
import TranscriptionStatus from '../components/TranscriptionStatus'
import useTranscriptionJob from '../components/useTranscriptionJob'
import { Alert, PageHeader, ProgressSteps } from '../components/UI'

export default function CreateTranscriptPage({ type = 'LECTURE' }) {
  const isNote = type === 'NOTE'
  const [title, setTitle] = useState('')
  const [file, setFile] = useState(null)
  const [errors, setErrors] = useState({})
  const [inputMode, setInputMode] = useState('capture')
  const { job, start, reset } = useTranscriptionJob()
  const navigate = useNavigate()
  useEffect(() => {
    if (job?.status === 'COMPLETED' && job.transcriptId) {
      const timer = window.setTimeout(() => navigate(`/transcripts/${job.transcriptId}`), 550)
      return () => window.clearTimeout(timer)
    }
  }, [job, navigate])
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
      <div className="page page--narrow">
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
    <div className="page">
      <PageHeader
        eyebrow={isNote ? 'Speak it. Save it. Study it.' : 'Accessible learning starts here'}
        title={isNote ? 'Create a self-study note' : 'Caption a recorded lecture'}
        description={
          isNote
            ? 'Record a thought or upload an audio clip. We’ll turn it into editable Sinhala text.'
            : 'Upload an audio or video recording and SinhaSpeech will create an editable Sinhala transcript.'
        }
      />
      <ProgressSteps
        steps={
          isNote
            ? ['Name your note', 'Record or upload', 'Review transcript']
            : ['Choose media', 'Process speech', 'Review transcript']
        }
        current={isNote ? (title ? 1 : 0) : 0}
      />
      <section className="form-card">
        <div className="form-card__heading">
          <span>1</span>
          <div>
            <h2>{isNote ? 'Name your note' : 'Lecture details'}</h2>
            <p>Use a clear title so you can find it later.</p>
          </div>
        </div>
        <div className="field">
          <label htmlFor="media-title">{isNote ? 'Note title' : 'Lecture title'}</label>
          <input
            id="media-title"
            value={title}
            onChange={(e) => {
              setTitle(e.target.value)
              setErrors({ ...errors, title: '' })
            }}
            placeholder={
              isNote
                ? 'e.g. Database revision — Week 4'
                : 'e.g. Introduction to Algorithms — Week 2'
            }
            aria-invalid={Boolean(errors.title)}
          />
          {errors.title && (
            <span className="field-error" role="alert">
              {errors.title}
            </span>
          )}
        </div>
      </section>
      <section className="form-card">
        <div className="form-card__heading">
          <span>2</span>
          <div>
            <h2>{isNote ? 'Add your voice note' : 'Add the lecture recording'}</h2>
            <p>
              {isNote
                ? 'Record here or choose an audio clip you already have.'
                : 'Choose a common audio or video format.'}
            </p>
          </div>
        </div>
        {isNote ? (
          <>
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
          </>
        ) : (
          <>
            <FileUpload
              file={file}
              onChange={(value) => {
                setFile(value)
                setErrors({ ...errors, file: '' })
              }}
              error={errors.file}
            />
            <div className="button-row button-row--end">
              <button type="button" className="button button--primary" onClick={() => begin()}>
                <Icon name="upload" size={17} /> Upload & transcribe
              </button>
            </div>
          </>
        )}
      </section>
      <Alert type="info" title="Your privacy matters">
        Recordings are stored securely for transcription and later review. Only you and authorized
        course staff can access them. Do not upload sensitive personal information.
      </Alert>
    </div>
  )
}
